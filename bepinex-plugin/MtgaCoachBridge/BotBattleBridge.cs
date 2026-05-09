using System;
using System.Collections.Generic;
using System.Reflection;
using BepInEx.Logging;
using GreClient.Rules;
using Newtonsoft.Json.Linq;
using UnityEngine;
using Wotc.Mtga.Cards.Database;

namespace MtgaCoachBridge
{
    // Phase 1 stub: an IHeadlessClientStrategy that delegates every request
    // to GoldfishStrategy and logs the type + outcome. Used to smoke-test
    // that the bot-battle plumbing works end-to-end before wiring the
    // strategy to the Python coach over a second named pipe.
    internal class BridgeStrategy : IHeadlessClientStrategy
    {
        private readonly IHeadlessClientStrategy _inner;
        private readonly string _label;
        private readonly ManualLogSource _log;
        private int _requestCount;

        public BridgeStrategy(string label, ManualLogSource log)
        {
            _label = label ?? "bridge";
            _log = log;
            // GoldfishStrategy in the ILSpy output is synthetic — the real
            // SharedClientCore.dll exports RequestHandlerStrategy +
            // GoldfishRequestHandlerFactory. Use the factory directly.
            _inner = new RequestHandlerStrategy(new GoldfishRequestHandlerFactory());
        }

        public int RequestCount => _requestCount;

        public void HandleRequest(BaseUserRequest request)
        {
            _requestCount++;
            var t = request != null ? request.GetType().Name : "<null>";
            _log?.LogInfo($"[Bridge:{_label}] #{_requestCount} HandleRequest: {t}");
            try
            {
                _inner.HandleRequest(request);
            }
            catch (Exception ex)
            {
                _log?.LogError($"[Bridge:{_label}] inner.HandleRequest threw: {ex}");
                throw;
            }
        }

        public void SetGameState(MtgGameState state)
        {
            _inner.SetGameState(state);
        }
    }

    internal static class BotBattleBridge
    {
        // Latest run state — exposed via bot_battle_status pipe command. The
        // start_bot_battle handler writes here from the Unity main thread and
        // the pipe-thread handler reads (locked).
        private static readonly object _stateLock = new object();
        private static string _lastError;
        private static bool _running;
        private static int _matchesRequested;
        private static int _matchesCompleted;
        private static BridgeStrategy _localStrategy;
        private static BridgeStrategy _opponentStrategy;

        public static JObject GetStatus()
        {
            lock (_stateLock)
            {
                return new JObject
                {
                    ["running"] = _running,
                    ["matches_requested"] = _matchesRequested,
                    ["matches_completed"] = _matchesCompleted,
                    ["local_request_count"] = _localStrategy != null ? _localStrategy.RequestCount : 0,
                    ["opponent_request_count"] = _opponentStrategy != null ? _opponentStrategy.RequestCount : 0,
                    ["last_error"] = _lastError ?? string.Empty,
                };
            }
        }

        public static JObject Start(JObject cfgJson, ManualLogSource log)
        {
            try
            {
                int matches = cfgJson.Value<int?>("matches") ?? 1;
                string sets = cfgJson.Value<string>("sets") ?? string.Empty; // e.g. "EOE,TDM"

                lock (_stateLock)
                {
                    if (_running)
                    {
                        return new JObject { ["ok"] = false, ["error"] = "bot battle already running" };
                    }
                    _running = true;
                    _lastError = null;
                    _matchesRequested = matches;
                    _matchesCompleted = 0;
                }

                // Build a random deck for each side. We need a CardDatabase to
                // resolve printings; the existing dev config loads one via
                // BotBattleConfigView.GetLocalCardDatabase but we need to do
                // the equivalent from BepInEx. PAPA owns the live db.
                var papa = UnityEngine.Object.FindObjectOfType<PAPA>();
                if (papa == null)
                {
                    return Fail(log, "PAPA not found — is MTGA past the title screen?");
                }
                var cardDb = papa.CardDatabase;
                if (cardDb == null)
                {
                    return Fail(log, "PAPA.CardDatabase is null — db not loaded yet?");
                }

                List<uint> localDeck;
                List<uint> opponentDeck;
                try
                {
                    localDeck = BotBattleConfig_DeckTest.GenerateRandomDeckFromSets(cardDb.DatabaseUtilities, sets);
                    opponentDeck = BotBattleConfig_DeckTest.GenerateRandomDeckFromSets(cardDb.DatabaseUtilities, sets);
                }
                catch (Exception ex)
                {
                    return Fail(log, $"deck generation failed: {ex.Message}");
                }
                if (localDeck == null || localDeck.Count == 0 || opponentDeck == null || opponentDeck.Count == 0)
                {
                    return Fail(log, $"empty deck (sets='{sets}'); try a known set code");
                }

                var dsConfig = new BotBattleDSConfig
                {
                    SessionType = BotBattleSessionType.DeckTest,
                    MatchesToPlay = matches,
                    LocalPlayerStrategy = BotBattleStrategyType.Goldfish,
                    OpponentStrategy = BotBattleStrategyType.Goldfish,
                    LocalPlayerCardsToTest = new List<List<uint>> { localDeck },
                    OpponentCardsToTest = new List<List<uint>> { opponentDeck },
                };

                log?.LogInfo($"[BotBattleBridge] dispatching BotBattleScene.Load: matches={matches} sets='{sets}' localDeck={localDeck.Count} oppDeck={opponentDeck.Count}");

                // Hand off to BotBattleScene's static loader. After it
                // enqueues the test, we hot-swap the strategy via reflection
                // (next match cycle) so the bridge gets the requests instead
                // of GoldfishStrategy.
                BotBattleScene.Load(dsConfig);

                // Defer the strategy hot-swap one frame: BotBattleScene needs
                // its scene to load + EnqueueTests to run before _testQueue
                // contains anything we can mutate. The host's Update tick is
                // the natural cadence.
                MtgaCoachHost.Instance?.StartCoroutine(SwapStrategiesNextFrames(log));

                return new JObject
                {
                    ["ok"] = true,
                    ["matches"] = matches,
                    ["local_deck_size"] = localDeck.Count,
                    ["opponent_deck_size"] = opponentDeck.Count,
                };
            }
            catch (Exception ex)
            {
                return Fail(log, $"unhandled: {ex.Message}");
            }
        }

        private static JObject Fail(ManualLogSource log, string msg)
        {
            log?.LogWarning($"[BotBattleBridge] {msg}");
            lock (_stateLock)
            {
                _lastError = msg;
                _running = false;
            }
            return new JObject { ["ok"] = false, ["error"] = msg };
        }

        // Yield-based coroutine: poll BotBattleScene for the test queue and
        // swap LocalPlayerStrategy / OpponentStrategy on the dequeued test.
        // _testQueue and _currentTest are private — reach via reflection.
        private static System.Collections.IEnumerator SwapStrategiesNextFrames(ManualLogSource log)
        {
            const float maxWaitSec = 30f;
            float elapsed = 0f;
            BotBattleScene scene = null;
            while (elapsed < maxWaitSec)
            {
                yield return null;
                elapsed += Time.unscaledDeltaTime;
                scene = UnityEngine.Object.FindObjectOfType<BotBattleScene>();
                if (scene != null) break;
            }
            if (scene == null)
            {
                Fail(log, "BotBattleScene didn't appear within 30s — scene transition blocked?");
                yield break;
            }

            // Wait for _currentTest to be assigned (RunTest dequeues + sets)
            var sceneType = typeof(BotBattleScene);
            var currentTestField = sceneType.GetField("_currentTest", BindingFlags.NonPublic | BindingFlags.Instance);
            if (currentTestField == null)
            {
                Fail(log, "_currentTest field not found via reflection");
                yield break;
            }

            float swapWait = 0f;
            BotBattleTest currentTest = null;
            while (swapWait < maxWaitSec)
            {
                yield return null;
                swapWait += Time.unscaledDeltaTime;
                currentTest = currentTestField.GetValue(scene) as BotBattleTest;
                if (currentTest != null) break;
            }
            if (currentTest == null)
            {
                Fail(log, "_currentTest stayed null — test never started");
                yield break;
            }

            var local = new BridgeStrategy("local", log);
            var opp = new BridgeStrategy("opp", log);
            currentTest.LocalPlayerStrategy = local;
            currentTest.OpponentStrategy = opp;
            lock (_stateLock)
            {
                _localStrategy = local;
                _opponentStrategy = opp;
            }
            log?.LogInfo("[BotBattleBridge] strategies swapped on _currentTest");
        }
    }
}
