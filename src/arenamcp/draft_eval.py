"""Draft card evaluation logic shared between MCP server and standalone.

This module contains the composite scoring logic for draft picks, combining:
- 17lands GIH win rate data
- Card type/mechanic value scoring
- On-color bonus calculation
- Synergy detection with picked cards
"""

from dataclasses import dataclass
from typing import Optional

from arenamcp.scryfall import ScryfallCache, ScryfallCard
from arenamcp.draftstats import DraftStatsCache
from arenamcp.mtgadb import MTGADatabase


@dataclass
class CardEvaluation:
    """Evaluation result for a draft pick candidate."""
    grp_id: int
    name: str
    score: float
    gih_wr: Optional[float]
    reason: str
    all_reasons: list[str]


def get_deck_colors(
    picked_cards: list[int],
    scryfall: ScryfallCache
) -> set[str]:
    """Determine deck colors from picked cards.

    Args:
        picked_cards: List of grp_ids already picked
        scryfall: Scryfall cache for card lookups

    Returns:
        Set of color letters (W, U, B, R, G) in the deck
    """
    colors = set()
    for grp_id in picked_cards:
        card = scryfall.get_card_by_arena_id(grp_id)
        if card and card.colors:
            colors.update(card.colors)
    return colors


def get_card_type_score(type_line: str, oracle_text: str) -> tuple[float, str]:
    """Score card by type and mechanics.

    Args:
        type_line: Card type line (e.g., "Creature - Human Wizard")
        oracle_text: Card oracle text

    Returns:
        Tuple of (score, reason) where score is 0-20 and reason explains it
    """
    oracle_lower = oracle_text.lower() if oracle_text else ""
    type_lower = type_line.lower() if type_line else ""

    # Removal detection
    removal_words = ["destroy", "exile", "damage", "fights", "-x/-x", "murder", "kill"]
    if any(word in oracle_lower for word in removal_words) and "creature" in oracle_lower:
        return (15.0, "removal")

    # Card draw
    if "draw" in oracle_lower and "card" in oracle_lower:
        return (10.0, "card draw")

    # Bombs (planeswalkers, big effects)
    if "planeswalker" in type_lower:
        return (20.0, "planeswalker")

    # Evasion
    if any(word in oracle_lower for word in ["flying", "menace", "trample", "unblockable"]):
        return (8.0, "evasion")

    # Creatures are decent baseline
    if "creature" in type_lower:
        return (5.0, "creature")

    # Lands
    if "land" in type_lower and "basic" not in type_lower:
        return (3.0, "fixing")

    return (0.0, "")


def check_synergy(
    card: ScryfallCard,
    picked_cards: list[int],
    scryfall: ScryfallCache
) -> tuple[float, str]:
    """Check for synergies with picked cards.

    Args:
        card: Card to evaluate
        picked_cards: List of grp_ids already picked
        scryfall: Scryfall cache for card lookups

    Returns:
        Tuple of (bonus_score, reason) for synergy bonus
    """
    if not picked_cards or not card:
        return (0.0, "")

    card_oracle = (card.oracle_text or "").lower()
    card_types = (card.type_line or "").lower()

    # Check last 5 picks for speed
    for grp_id in picked_cards[-5:]:
        picked = scryfall.get_card_by_arena_id(grp_id)
        if not picked:
            continue

        picked_oracle = (picked.oracle_text or "").lower()
        picked_name = picked.name.lower()

        # Direct name reference
        if picked_name in card_oracle:
            return (12.0, f"synergy with {picked.name}")

        # Tribal synergy
        creature_types = [
            "goblin", "elf", "merfolk", "zombie", "vampire",
            "human", "wizard", "warrior", "eldrazi"
        ]
        for tribe in creature_types:
            if tribe in card_types and tribe in picked_oracle:
                return (8.0, f"{tribe} synergy")

        # Mechanic synergy keywords
        mechanics = [
            "energy", "adapt", "proliferate", "counter",
            "token", "graveyard", "sacrifice"
        ]
        for mech in mechanics:
            if mech in card_oracle and mech in picked_oracle:
                return (6.0, f"{mech} synergy")

    return (0.0, "")


def evaluate_pack(
    cards_in_pack: list[int],
    picked_cards: list[int],
    set_code: str,
    scryfall: ScryfallCache,
    draft_stats: Optional[DraftStatsCache] = None,
    mtgadb: Optional[MTGADatabase] = None,
) -> list[CardEvaluation]:
    """Evaluate all cards in a pack with composite scoring.

    Args:
        cards_in_pack: List of grp_ids in current pack
        picked_cards: List of grp_ids already picked
        set_code: Set code for 17lands lookup (e.g., "MH3")
        scryfall: Scryfall cache for card data
        draft_stats: Optional 17lands stats cache
        mtgadb: Optional MTGA database for card names

    Returns:
        List of CardEvaluation sorted by score (highest first)
    """
    deck_colors = get_deck_colors(picked_cards, scryfall)
    evaluations = []

    for grp_id in cards_in_pack:
        # Try to get card data from Scryfall
        card = scryfall.get_card_by_arena_id(grp_id)

        # Fall back to MTGA database for name if Scryfall fails
        if not card and mtgadb and mtgadb.available:
            mtga_card = mtgadb.get_card(grp_id)
            if mtga_card:
                # Create a minimal card-like object for evaluation
                # (won't have oracle text but will have name)
                card_name = mtga_card.name
            else:
                continue
        elif not card:
            continue
        else:
            card_name = card.name

        score = 0.0
        reasons = []

        # 17lands GIH win rate (scaled to 0-100)
        gih_wr = None
        if set_code and draft_stats:
            stats = draft_stats.get_draft_rating(card_name, set_code)
            if stats and stats.gih_wr:
                gih_wr = stats.gih_wr
                score += gih_wr * 100  # 0-100 scale
                reasons.append(f"{int(gih_wr * 100)}% WR")

        # Card type/mechanic value (fallback when no 17lands)
        if card:
            type_score, type_reason = get_card_type_score(
                card.type_line, card.oracle_text
            )
            if not gih_wr:
                score += type_score + 40  # Base score without 17lands
            if type_reason and type_reason != "creature":
                reasons.append(type_reason)

            # On-color bonus
            card_colors = set(card.colors) if card.colors else set()
            if deck_colors and card_colors:
                if card_colors.issubset(deck_colors):
                    score += 12.0
                    reasons.append("on color")
                elif card_colors & deck_colors:
                    score += 6.0
                    reasons.append("splashable")
            elif not card_colors:  # Colorless fits any deck
                score += 4.0

            # Synergy bonus
            syn_score, syn_reason = check_synergy(card, picked_cards, scryfall)
            if syn_score:
                score += syn_score
                reasons.append(syn_reason)

        # Pick best reason (most specific)
        best_reason = reasons[-1] if reasons else ""

        evaluations.append(CardEvaluation(
            grp_id=grp_id,
            name=card_name,
            score=score,
            gih_wr=gih_wr,
            reason=best_reason,
            all_reasons=reasons,
        ))

    # Sort by score descending
    evaluations.sort(key=lambda e: e.score, reverse=True)
    return evaluations


def format_pick_recommendation(
    evaluations: list[CardEvaluation],
    pack_number: int,
    pick_number: int,
    num_recommendations: int = 2
) -> str:
    """Format spoken recommendation for top picks.

    Args:
        evaluations: Evaluated cards sorted by score
        pack_number: Current pack number (1-3)
        pick_number: Current pick in pack
        num_recommendations: How many picks to recommend

    Returns:
        Human-readable recommendation string for TTS
    """
    if not evaluations:
        return f"Pack {pack_number}, Pick {pick_number}. No cards found."

    if len(evaluations) == 1:
        top = evaluations[0]
        reason = f", {top.reason}" if top.reason else ""
        return f"Pack {pack_number}, Pick {pick_number}. Take {top.name}{reason}."

    pack_pick = f"Pack {pack_number}, Pick {pick_number}."

    top1 = evaluations[0]
    top2 = evaluations[1] if len(evaluations) > 1 else None

    r1 = f", {top1.reason}" if top1.reason else ""

    if top2:
        r2 = f", {top2.reason}" if top2.reason else ""
        return f"{pack_pick} Take {top1.name}{r1}. Or {top2.name}{r2}."
    else:
        return f"{pack_pick} Take {top1.name}{r1}."
