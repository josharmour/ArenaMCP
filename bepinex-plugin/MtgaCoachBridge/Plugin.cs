using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Reflection;
using System.Text;
using System.Threading;
using BepInEx;
using BepInEx.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using GreClient.Rules;
using Wotc.Mtgo.Gre.External.Messaging;

namespace MtgaCoachBridge
{
    [BepInPlugin(PluginInfo.GUID, PluginInfo.Name, PluginInfo.Version)]
    public class Plugin : BaseUnityPlugin
    {
        private static ManualLogSource _log;
        private Thread _pipeThread;
        private volatile bool _running;
        private readonly ConcurrentQueue<PipeCommand> _pendingCommands = new ConcurrentQueue<PipeCommand>();

        private BaseUserRequest _lastKnownRequest;
        private readonly object _interactionLock = new object();

        // Cached reference to GameManager (only valid on main thread)
        private GameManager _cachedGameManager;
        private float _lastGameManagerLookup;

        private void Awake()
        {
            _log = Logger;
            _log.LogInfo($"MtgaCoachBridge v{PluginInfo.Version} loaded");

            _running = true;
            _pipeThread = new Thread(PipeServerLoop)
            {
                IsBackground = true,
                Name = "MtgaCoachBridge-Pipe"
            };
            _pipeThread.Start();
        }

        private void OnDestroy()
        {
            _running = false;
        }

        private void Update()
        {
            while (_pendingCommands.TryDequeue(out var cmd))
            {
                try
                {
                    ProcessCommand(cmd);
                }
                catch (Exception ex)
                {
                    _log.LogError($"Error processing command: {ex}");
                    cmd.SetResponse(new JObject
                    {
                        ["ok"] = false,
                        ["error"] = ex.Message
                    });
                }
            }
        }

        // -------------------------------------------------------------------
        // Named pipe server
        // -------------------------------------------------------------------

        private void PipeServerLoop()
        {
            while (_running)
            {
                NamedPipeServerStream pipe = null;
                try
                {
                    pipe = new NamedPipeServerStream(
                        "mtgacoach_gre",
                        PipeDirection.InOut,
                        1,
                        PipeTransmissionMode.Byte,
                        PipeOptions.Asynchronous
                    );

                    _log.LogInfo("Pipe server waiting for connection on \\\\.\\pipe\\mtgacoach_gre");
                    pipe.WaitForConnection();
                    _log.LogInfo("Pipe client connected");

                    HandleClient(pipe);
                }
                catch (Exception ex)
                {
                    if (_running)
                        _log.LogWarning($"Pipe error: {ex.Message}");
                }
                finally
                {
                    try { pipe?.Dispose(); } catch { }
                }

                if (_running)
                    Thread.Sleep(500);
            }
        }

        private void HandleClient(NamedPipeServerStream pipe)
        {
            using var reader = new StreamReader(pipe, Encoding.UTF8, false, 4096, leaveOpen: true);
            using var writer = new StreamWriter(pipe, Encoding.UTF8, 4096, leaveOpen: true)
            {
                AutoFlush = true
            };

            while (_running && pipe.IsConnected)
            {
                string line;
                try
                {
                    line = reader.ReadLine();
                }
                catch
                {
                    break;
                }

                if (line == null)
                    break;

                line = line.Trim();
                if (string.IsNullOrEmpty(line))
                    continue;

                try
                {
                    var json = JObject.Parse(line);
                    var cmd = new PipeCommand(json);
                    _pendingCommands.Enqueue(cmd);

                    var response = cmd.WaitForResponse(5000);
                    writer.WriteLine(response.ToString(Formatting.None));
                }
                catch (Exception ex)
                {
                    var errorResp = new JObject
                    {
                        ["ok"] = false,
                        ["error"] = $"Parse error: {ex.Message}"
                    };
                    try { writer.WriteLine(errorResp.ToString(Formatting.None)); } catch { break; }
                }
            }

            _log.LogInfo("Pipe client disconnected");
        }

        // -------------------------------------------------------------------
        // GameManager access (cached, main thread only)
        // -------------------------------------------------------------------

        private GameManager GetGameManager()
        {
            float now = Time.unscaledTime;
            if (_cachedGameManager == null || now - _lastGameManagerLookup > 5f)
            {
                _cachedGameManager = FindObjectOfType<GameManager>();
                _lastGameManagerLookup = now;
            }
            return _cachedGameManager;
        }

        // -------------------------------------------------------------------
        // Command processing (runs on Unity main thread)
        // -------------------------------------------------------------------

        private void ProcessCommand(PipeCommand cmd)
        {
            string action = cmd.Json.Value<string>("action") ?? "";

            switch (action)
            {
                case "ping":
                    cmd.SetResponse(new JObject
                    {
                        ["ok"] = true,
                        ["version"] = PluginInfo.Version
                    });
                    break;

                case "get_pending_actions":
                    HandleGetPendingActions(cmd);
                    break;

                case "submit_action":
                    HandleSubmitAction(cmd);
                    break;

                case "submit_pass":
                    HandleSubmitPass(cmd);
                    break;

                case "get_game_state":
                    HandleGetGameState(cmd);
                    break;

                case "get_timer_state":
                    HandleGetTimerState(cmd);
                    break;

                case "get_match_info":
                    HandleGetMatchInfo(cmd);
                    break;

                default:
                    cmd.SetResponse(new JObject
                    {
                        ["ok"] = false,
                        ["error"] = $"Unknown action: {action}"
                    });
                    break;
            }
        }

        // -------------------------------------------------------------------
        // Find pending interaction
        // -------------------------------------------------------------------

        private BaseUserRequest FindPendingInteraction()
        {
            try
            {
                var gameManager = GetGameManager();
                if (gameManager == null)
                {
                    _log.LogDebug("GameManager not found in scene");
                    return null;
                }

                var wfc = gameManager.WorkflowController;
                if (wfc == null)
                {
                    _log.LogDebug("WorkflowController is null");
                    return null;
                }

                object workflow = wfc.CurrentWorkflow;
                if (workflow == null)
                {
                    try
                    {
                        var pendingProp = wfc.GetType().GetProperty("PendingWorkflow",
                            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                        if (pendingProp != null)
                            workflow = pendingProp.GetValue(wfc);
                    }
                    catch { }
                }

                if (workflow == null)
                {
                    _log.LogDebug("No current or pending workflow");
                    return null;
                }

                var reqProp = workflow.GetType().GetProperty("BaseRequest",
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (reqProp == null)
                {
                    reqProp = workflow.GetType().GetProperty("Request",
                        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                }
                if (reqProp != null)
                {
                    var request = reqProp.GetValue(workflow) as BaseUserRequest;
                    if (request != null)
                    {
                        _log.LogDebug($"Found pending request: {request.GetType().Name}");
                        return request;
                    }
                }

                // Fallback: MatchManager reflection
                try
                {
                    var mm = gameManager.MatchManager;
                    if (mm != null)
                    {
                        var field = mm.GetType().GetField("_pendingInteraction",
                            BindingFlags.NonPublic | BindingFlags.Instance);
                        if (field != null)
                        {
                            var request = field.GetValue(mm) as BaseUserRequest;
                            if (request != null)
                                return request;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _log.LogDebug($"MatchManager fallback: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                _log.LogDebug($"FindPendingInteraction error: {ex.Message}");
            }

            return null;
        }

        // -------------------------------------------------------------------
        // Existing commands: get_pending_actions, submit_action, submit_pass
        // -------------------------------------------------------------------

        private void HandleGetPendingActions(PipeCommand cmd)
        {
            var request = FindPendingInteraction();
            if (request == null)
            {
                cmd.SetResponse(new JObject
                {
                    ["ok"] = true,
                    ["has_pending"] = false,
                    ["request_type"] = JValue.CreateNull()
                });
                return;
            }

            lock (_interactionLock)
            {
                _lastKnownRequest = request;
            }

            var resp = new JObject
            {
                ["ok"] = true,
                ["has_pending"] = true,
                ["request_type"] = request.Type.ToString(),
                ["can_cancel"] = request.CanCancel,
                ["allow_undo"] = request.AllowUndo
            };

            if (request is ActionsAvailableRequest actionsReq)
            {
                var actionsArr = new JArray();
                for (int i = 0; i < actionsReq.Actions.Count; i++)
                {
                    actionsArr.Add(SerializeAction(actionsReq.Actions[i]));
                }
                resp["actions"] = actionsArr;
                resp["can_pass"] = actionsReq.CanPass;
            }

            cmd.SetResponse(resp);
        }

        private void HandleSubmitAction(PipeCommand cmd)
        {
            BaseUserRequest request;
            lock (_interactionLock)
            {
                request = _lastKnownRequest;
            }

            if (request == null)
                request = FindPendingInteraction();

            if (request == null)
            {
                cmd.SetResponse(new JObject
                {
                    ["ok"] = false,
                    ["error"] = "No pending interaction"
                });
                return;
            }

            if (request is ActionsAvailableRequest actionsReq)
            {
                int actionIndex = cmd.Json.Value<int>("action_index");
                bool autoPass = cmd.Json.Value<bool>("auto_pass");

                if (actionIndex < 0 || actionIndex >= actionsReq.Actions.Count)
                {
                    cmd.SetResponse(new JObject
                    {
                        ["ok"] = false,
                        ["error"] = $"Action index {actionIndex} out of range (0-{actionsReq.Actions.Count - 1})"
                    });
                    return;
                }

                var action = actionsReq.Actions[actionIndex];
                _log.LogInfo($"Submitting action [{actionIndex}]: {action.ActionType} grpId={action.GrpId} instanceId={action.InstanceId}");

                actionsReq.SubmitAction(action, autoPass);

                lock (_interactionLock)
                {
                    _lastKnownRequest = null;
                }

                cmd.SetResponse(new JObject
                {
                    ["ok"] = true,
                    ["submitted_type"] = action.ActionType.ToString(),
                    ["submitted_grp_id"] = (int)action.GrpId,
                    ["submitted_instance_id"] = (int)action.InstanceId
                });
            }
            else
            {
                cmd.SetResponse(new JObject
                {
                    ["ok"] = false,
                    ["error"] = $"Pending request is {request.GetType().Name}, not ActionsAvailableRequest"
                });
            }
        }

        private void HandleSubmitPass(PipeCommand cmd)
        {
            BaseUserRequest request;
            lock (_interactionLock)
            {
                request = _lastKnownRequest;
            }

            if (request == null)
                request = FindPendingInteraction();

            if (request == null)
            {
                cmd.SetResponse(new JObject
                {
                    ["ok"] = false,
                    ["error"] = "No pending interaction"
                });
                return;
            }

            if (request is ActionsAvailableRequest actionsReq && actionsReq.CanPass)
            {
                _log.LogInfo("Submitting pass");
                actionsReq.SubmitPass();

                lock (_interactionLock)
                {
                    _lastKnownRequest = null;
                }

                cmd.SetResponse(new JObject
                {
                    ["ok"] = true,
                    ["submitted_type"] = "Pass"
                });
            }
            else
            {
                cmd.SetResponse(new JObject
                {
                    ["ok"] = false,
                    ["error"] = "Cannot pass on current interaction"
                });
            }
        }

        // -------------------------------------------------------------------
        // Phase 2: get_game_state — full game state from MtgGameState
        // -------------------------------------------------------------------

        private void HandleGetGameState(PipeCommand cmd)
        {
            var gm = GetGameManager();
            if (gm == null)
            {
                cmd.SetResponse(new JObject { ["ok"] = false, ["error"] = "GameManager not found" });
                return;
            }

            MtgGameState gs;
            try
            {
                gs = gm.CurrentGameState;
            }
            catch (Exception ex)
            {
                cmd.SetResponse(new JObject { ["ok"] = false, ["error"] = $"CurrentGameState error: {ex.Message}" });
                return;
            }

            if (gs == null)
            {
                cmd.SetResponse(new JObject { ["ok"] = false, ["error"] = "No active game state" });
                return;
            }

            try
            {
                var resp = new JObject { ["ok"] = true };

                // Turn info
                resp["turn"] = new JObject
                {
                    ["turn_number"] = gs.GameWideTurn,
                    ["phase"] = gs.CurrentPhase.ToString(),
                    ["step"] = gs.CurrentStep.ToString(),
                    ["active_player"] = gs.ActivePlayer?.ControllerId ?? 0,
                    ["deciding_player"] = gs.DecidingPlayer?.ControllerId ?? 0,
                    ["stage"] = gs.Stage.ToString(),
                };

                // Players
                var playersArr = new JArray();
                if (gs.Players != null)
                {
                    foreach (var p in gs.Players)
                    {
                        var pObj = new JObject
                        {
                            ["seat_id"] = p.ControllerId,
                            ["life_total"] = p.LifeTotal,
                            ["is_local"] = p.IsLocalPlayer,
                            ["status"] = p.Status.ToString(),
                            ["mulligan_count"] = p.MulliganCount,
                            ["timeout_count"] = p.TimeoutCount,
                        };
                        // Mana pool
                        if (p.ManaPool != null && p.ManaPool.Count > 0)
                        {
                            var mana = new JObject();
                            foreach (var m in p.ManaPool)
                            {
                                string color = m.Color.ToString();
                                int current = mana[color]?.Value<int>() ?? 0;
                                mana[color] = current + (int)m.Count;
                            }
                            pObj["mana_pool"] = mana;
                        }
                        // Commander IDs
                        if (p.CommanderIds != null && p.CommanderIds.Count > 0)
                        {
                            var cmdIds = new JArray();
                            foreach (var cid in p.CommanderIds)
                                cmdIds.Add((int)cid);
                            pObj["commander_ids"] = cmdIds;
                        }
                        // Dungeon
                        if (p.DungeonState != null)
                        {
                            pObj["dungeon"] = new JObject
                            {
                                ["dungeon_id"] = (int)(p.DungeonState.DungeonId ?? 0),
                                ["room_id"] = (int)(p.DungeonState.RoomId ?? 0),
                            };
                        }
                        // Designations (monarch, initiative, etc.)
                        if (p.Designations != null && p.Designations.Count > 0)
                        {
                            var desigs = new JArray();
                            foreach (var d in p.Designations)
                                desigs.Add(d.Type.ToString());
                            pObj["designations"] = desigs;
                        }
                        playersArr.Add(pObj);
                    }
                }
                resp["players"] = playersArr;

                // Zones with card instances
                resp["zones"] = SerializeZones(gs);

                // Combat info
                if (gs.AttackInfo != null && gs.AttackInfo.Count > 0)
                {
                    var attacks = new JObject();
                    foreach (var kvp in gs.AttackInfo)
                        attacks[kvp.Key.ToString()] = kvp.Value.TargetId.ToString();
                    resp["attack_info"] = attacks;
                }
                if (gs.BlockInfo != null && gs.BlockInfo.Count > 0)
                {
                    var blocks = new JObject();
                    foreach (var kvp in gs.BlockInfo)
                        blocks[kvp.Key.ToString()] = kvp.Value.AttackerId.ToString();
                    resp["block_info"] = blocks;
                }

                // Designations (game-level)
                if (gs.Designations != null && gs.Designations.Count > 0)
                {
                    var desigs = new JArray();
                    foreach (var d in gs.Designations)
                    {
                        desigs.Add(new JObject
                        {
                            ["type"] = d.Type.ToString(),
                            ["player_id"] = (int)d.PlayerId,
                        });
                    }
                    resp["designations"] = desigs;
                }

                // Timers
                if (gs.Timers != null && gs.Timers.Count > 0)
                {
                    resp["timers"] = SerializeTimers(gs.Timers);
                }

                // Pending interaction type
                var pending = FindPendingInteraction();
                if (pending != null)
                {
                    resp["pending_interaction"] = pending.GetType().Name;
                }

                cmd.SetResponse(resp);
            }
            catch (Exception ex)
            {
                _log.LogError($"get_game_state serialization error: {ex}");
                cmd.SetResponse(new JObject { ["ok"] = false, ["error"] = $"Serialization error: {ex.Message}" });
            }
        }

        private JObject SerializeZones(MtgGameState gs)
        {
            var zones = new JObject();

            void AddZone(string name, MtgZone zone)
            {
                if (zone == null) return;
                var cards = new JArray();
                if (zone.VisibleCards != null)
                {
                    foreach (var card in zone.VisibleCards)
                    {
                        cards.Add(SerializeCard(card));
                    }
                }
                zones[name] = new JObject
                {
                    ["zone_id"] = (int)zone.Id,
                    ["total_count"] = (int)zone.TotalCardCount,
                    ["cards"] = cards,
                };
            }

            try { AddZone("battlefield", gs.Battlefield); } catch { }
            try { AddZone("stack", gs.Stack); } catch { }
            try { AddZone("local_hand", gs.LocalHand); } catch { }
            try { AddZone("opponent_hand", gs.OpponentHand); } catch { }
            try { AddZone("local_graveyard", gs.LocalGraveyard); } catch { }
            try { AddZone("opponent_graveyard", gs.OpponentGraveyard); } catch { }
            try { AddZone("exile", gs.Exile); } catch { }
            try { AddZone("command", gs.Command); } catch { }
            try { AddZone("local_library", gs.LocalLibrary); } catch { }
            try { AddZone("opponent_library", gs.OpponentLibrary); } catch { }

            return zones;
        }

        private static JObject SerializeCard(MtgCardInstance card)
        {
            var obj = new JObject
            {
                ["instance_id"] = (int)card.InstanceId,
                ["grp_id"] = (int)card.GrpId,
                ["object_type"] = card.ObjectType.ToString(),
                ["is_tapped"] = card.IsTapped,
                ["owner_id"] = card.Owner?.ControllerId ?? 0,
                ["controller_id"] = card.Controller?.ControllerId ?? 0,
            };

            // Power/toughness
            try
            {
                if (card.Power != null)
                    obj["power"] = card.Power.Value;
                if (card.Toughness != null)
                    obj["toughness"] = card.Toughness.Value;
            }
            catch { }

            // Loyalty / Defense
            if (card.Loyalty.HasValue)
                obj["loyalty"] = (int)card.Loyalty.Value;
            if (card.Defense.HasValue)
                obj["defense"] = (int)card.Defense.Value;

            // Combat state
            if (card.IsAttacking)
            {
                obj["is_attacking"] = true;
                if (card.AttackTargetId != 0)
                    obj["attack_target_id"] = (int)card.AttackTargetId;
            }
            if (card.IsBlocking)
                obj["is_blocking"] = true;

            // Summoning sickness
            if (card.HasSummoningSickness)
                obj["summoning_sickness"] = true;

            // Phased out
            if (card.IsPhasedOut)
                obj["is_phased_out"] = true;

            // Damaged
            if (card.Damage > 0)
                obj["damage"] = (int)card.Damage;
            if (card.IsDamagedThisTurn)
                obj["damaged_this_turn"] = true;

            // Class level
            if (card.ClassLevel > 0)
                obj["class_level"] = card.ClassLevel;

            // Copy info
            if (card.IsCopy && card.CopyObjectGrpId != 0)
                obj["copied_from_grp_id"] = (int)card.CopyObjectGrpId;

            // Card types
            if (card.CardTypes != null && card.CardTypes.Count > 0)
            {
                var types = new JArray();
                foreach (var ct in card.CardTypes)
                    types.Add(ct.ToString());
                obj["card_types"] = types;
            }

            // Subtypes
            if (card.Subtypes != null && card.Subtypes.Count > 0)
            {
                var subs = new JArray();
                foreach (var st in card.Subtypes)
                    subs.Add(st.ToString());
                obj["subtypes"] = subs;
            }

            // Colors
            if (card.Colors != null && card.Colors.Count > 0)
            {
                var colors = new JArray();
                foreach (var c in card.Colors)
                    colors.Add(c.ToString());
                obj["colors"] = colors;
            }

            // Counters
            if (card.Counters != null && card.Counters.Count > 0)
            {
                var counters = new JObject();
                foreach (var kvp in card.Counters)
                    counters[kvp.Key.ToString()] = kvp.Value;
                obj["counters"] = counters;
            }

            // Color production (mana abilities)
            if (card.ColorProduction != null && card.ColorProduction.Count > 0)
            {
                var cp = new JArray();
                foreach (var c in card.ColorProduction)
                    cp.Add(c.ToString());
                obj["color_production"] = cp;
            }

            // Targets
            if (card.TargetIds != null && card.TargetIds.Count > 0)
            {
                var tids = new JArray();
                foreach (var tid in card.TargetIds)
                    tids.Add((int)tid);
                obj["target_ids"] = tids;
            }

            // Attached to
            if (card.AttachedToId != 0)
                obj["attached_to_id"] = (int)card.AttachedToId;

            // Attached with (auras/equipment on this card)
            if (card.AttachedWithIds != null && card.AttachedWithIds.Count > 0)
            {
                var awIds = new JArray();
                foreach (var aid in card.AttachedWithIds)
                    awIds.Add((int)aid);
                obj["attached_with_ids"] = awIds;
            }

            // Revealed to opponent
            if (card.RevealedToOpponent)
                obj["revealed_to_opponent"] = true;

            // Face down
            if (card.FaceDownState != FaceDownState.None)
                obj["face_down"] = card.FaceDownState.ToString();

            // Crewed/saddled
            if (card.CrewedAndSaddledByIds != null && card.CrewedAndSaddledByIds.Count > 0)
                obj["crewed_this_turn"] = true;

            // Visibility
            obj["visibility"] = card.Visibility.ToString();

            return obj;
        }

        // -------------------------------------------------------------------
        // Phase 2: get_timer_state
        // -------------------------------------------------------------------

        private void HandleGetTimerState(PipeCommand cmd)
        {
            var gm = GetGameManager();
            if (gm == null)
            {
                cmd.SetResponse(new JObject { ["ok"] = false, ["error"] = "GameManager not found" });
                return;
            }

            var gs = gm.CurrentGameState;
            if (gs == null)
            {
                cmd.SetResponse(new JObject { ["ok"] = false, ["error"] = "No active game state" });
                return;
            }

            var resp = new JObject { ["ok"] = true };

            if (gs.Timers != null && gs.Timers.Count > 0)
            {
                resp["timers"] = SerializeTimers(gs.Timers);
            }

            // Also get per-player timers
            if (gs.Players != null)
            {
                var playerTimers = new JObject();
                foreach (var p in gs.Players)
                {
                    if (p.Timers != null && p.Timers.Count > 0)
                    {
                        var arr = new JArray();
                        foreach (var t in p.Timers)
                        {
                            arr.Add(new JObject
                            {
                                ["timer_id"] = (int)t.Id,
                                ["type"] = t.Type.ToString(),
                                ["duration_sec"] = (int)t.DurationSec,
                                ["elapsed_sec"] = (int)t.ElapsedSec,
                                ["running"] = t.Running,
                                ["behavior"] = t.Behavior.ToString(),
                            });
                        }
                        playerTimers[p.ControllerId.ToString()] = arr;
                    }
                }
                if (playerTimers.Count > 0)
                    resp["player_timers"] = playerTimers;
            }

            cmd.SetResponse(resp);
        }

        private static JObject SerializeTimers(Dictionary<uint, MtgTimer> timers)
        {
            var result = new JObject();
            foreach (var kvp in timers)
            {
                var t = kvp.Value;
                result[kvp.Key.ToString()] = new JObject
                {
                    ["timer_id"] = (int)t.Id,
                    ["type"] = t.Type.ToString(),
                    ["duration_sec"] = (int)t.DurationSec,
                    ["elapsed_sec"] = (int)t.ElapsedSec,
                    ["running"] = t.Running,
                    ["behavior"] = t.Behavior.ToString(),
                    ["warning_threshold"] = (int)t.WarningThresholdSec,
                };
            }
            return result;
        }

        // -------------------------------------------------------------------
        // Phase 2: get_match_info
        // -------------------------------------------------------------------

        private void HandleGetMatchInfo(PipeCommand cmd)
        {
            var gm = GetGameManager();
            if (gm == null)
            {
                cmd.SetResponse(new JObject { ["ok"] = false, ["error"] = "GameManager not found" });
                return;
            }

            var gs = gm.CurrentGameState;
            var resp = new JObject { ["ok"] = true };

            if (gs != null)
            {
                resp["game_state_id"] = gs.Id;
                resp["stage"] = gs.Stage.ToString();
                resp["turn"] = gs.GameWideTurn;
                resp["phase"] = gs.CurrentPhase.ToString();
                resp["step"] = gs.CurrentStep.ToString();

                if (gs.GameInfo != null)
                {
                    try
                    {
                        var gi = gs.GameInfo;
                        var info = new JObject();
                        // Use reflection to extract available fields
                        foreach (var prop in gi.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
                        {
                            try
                            {
                                var val = prop.GetValue(gi);
                                if (val != null)
                                    info[prop.Name] = val.ToString();
                            }
                            catch { }
                        }
                        resp["game_info"] = info;
                    }
                    catch { }
                }

                // Local/opponent info
                if (gs.LocalPlayer != null)
                {
                    resp["local_seat_id"] = gs.LocalPlayer.ControllerId;
                    resp["local_life"] = gs.LocalPlayer.LifeTotal;
                }
                if (gs.Opponent != null)
                {
                    resp["opponent_seat_id"] = gs.Opponent.ControllerId;
                    resp["opponent_life"] = gs.Opponent.LifeTotal;
                }
            }
            else
            {
                resp["stage"] = "no_game";
            }

            cmd.SetResponse(resp);
        }

        // -------------------------------------------------------------------
        // Enhanced action serialization (Phase 2)
        // -------------------------------------------------------------------

        private static JObject SerializeAction(Wotc.Mtgo.Gre.External.Messaging.Action action)
        {
            var obj = new JObject
            {
                ["actionType"] = action.ActionType.ToString(),
                ["grpId"] = (int)action.GrpId,
                ["instanceId"] = (int)action.InstanceId,
            };

            if (action.AbilityGrpId != 0)
                obj["abilityGrpId"] = (int)action.AbilityGrpId;
            if (action.SourceId != 0)
                obj["sourceId"] = (int)action.SourceId;
            if (action.AlternativeGrpId != 0)
                obj["alternativeGrpId"] = (int)action.AlternativeGrpId;
            if (action.FacetId != 0)
                obj["facetId"] = (int)action.FacetId;
            if (action.UniqueAbilityId != 0)
                obj["uniqueAbilityId"] = (int)action.UniqueAbilityId;

            // Castability flag from GRE
            obj["assumeCanBePaidFor"] = action.AssumeCanBePaidFor;

            // Mana cost
            if (action.ManaCost != null && action.ManaCost.Count > 0)
            {
                var costs = new JArray();
                for (int i = 0; i < action.ManaCost.Count; i++)
                {
                    var mc = action.ManaCost[i];
                    costs.Add(new JObject
                    {
                        ["color"] = mc.Color.ToString(),
                        ["count"] = (int)mc.Count
                    });
                }
                obj["manaCost"] = costs;
            }

            // Full AutoTap solution (Phase 2: serialize tap sequence, not just boolean)
            if (action.AutoTapSolution != null)
            {
                obj["hasAutoTap"] = true;
                try
                {
                    var ats = action.AutoTapSolution;
                    // AutoTapSolution has AutoTapActions — the lands to tap
                    var tapProp = ats.GetType().GetProperty("AutoTapActions")
                                  ?? ats.GetType().GetProperty("autoTapActions_");
                    if (tapProp != null)
                    {
                        var tapActions = tapProp.GetValue(ats) as System.Collections.IEnumerable;
                        if (tapActions != null)
                        {
                            var taps = new JArray();
                            foreach (var ta in tapActions)
                            {
                                var tapObj = new JObject();
                                // Extract instanceId and manaProduced via reflection
                                var instProp = ta.GetType().GetProperty("InstanceId");
                                var manaProp = ta.GetType().GetProperty("ManaId");
                                if (instProp != null)
                                    tapObj["instanceId"] = Convert.ToInt32(instProp.GetValue(ta));
                                if (manaProp != null)
                                    tapObj["manaId"] = Convert.ToInt32(manaProp.GetValue(ta));
                                taps.Add(tapObj);
                            }
                            if (taps.Count > 0)
                                obj["autoTapActions"] = taps;
                        }
                    }
                }
                catch (Exception ex)
                {
                    _log.LogDebug($"AutoTap serialization: {ex.Message}");
                }
            }

            // Targets on the action
            if (action.Targets != null && action.Targets.Count > 0)
            {
                var targets = new JArray();
                for (int i = 0; i < action.Targets.Count; i++)
                {
                    var t = action.Targets[i];
                    targets.Add(new JObject
                    {
                        ["targetId"] = (int)t.TargetIdx,
                    });
                }
                obj["targets"] = targets;
            }

            // Highlight (tells UI what to emphasize)
            if (action.Highlight != HighlightType.None)
                obj["highlight"] = action.Highlight.ToString();

            // ShouldStop flag
            if (action.ShouldStop)
                obj["shouldStop"] = true;

            // IsBatchable
            if (action.IsBatchable)
                obj["isBatchable"] = true;

            return obj;
        }
    }

    // -------------------------------------------------------------------
    // Helper: pipe command with synchronous response channel
    // -------------------------------------------------------------------

    internal class PipeCommand
    {
        public JObject Json { get; }
        private JObject _response;
        private readonly ManualResetEventSlim _signal = new ManualResetEventSlim(false);

        public PipeCommand(JObject json)
        {
            Json = json;
        }

        public void SetResponse(JObject response)
        {
            _response = response;
            _signal.Set();
        }

        public JObject WaitForResponse(int timeoutMs)
        {
            if (_signal.Wait(timeoutMs))
                return _response;

            return new JObject
            {
                ["ok"] = false,
                ["error"] = "Command timed out waiting for main thread"
            };
        }
    }

    internal static class PluginInfo
    {
        public const string GUID = "com.mtgacoach.grebridge";
        public const string Name = "MtgaCoach GRE Bridge";
        public const string Version = "0.2.0";
    }
}
