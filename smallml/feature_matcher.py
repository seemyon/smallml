"""
Feature Matcher for SmallML

Matches user feature names to pre-trained feature names using:
- Tier 1: Exact match (case-insensitive)
- Tier 2: Alias dictionary lookup

This enables transfer learning with arbitrary feature sets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FeatureMatcher:
    """
    Matches user features to pre-trained features using exact and alias matching.

    Parameters
    ----------
    pretrained_features : List[str]
        List of feature names from the pre-trained model
    aliases_path : Optional[Path]
        Path to JSON file with feature aliases. If None, uses default bundled aliases.

    Attributes
    ----------
    pretrained_features : List[str]
        Normalized (lowercase) pre-trained feature names
    aliases : Dict[str, List[str]]
        Dictionary mapping pre-trained features to their aliases
    reverse_alias_map : Dict[str, str]
        Reverse mapping from alias to pre-trained feature

    Examples
    --------
    >>> matcher = FeatureMatcher(pretrained_features=['recency', 'frequency', 'monetary'])
    >>> result = matcher.match('days_since_last_purchase')
    >>> print(result)
    ('recency', 1.0)

    >>> matches, info = matcher.match_all(['days_since_last_purchase', 'order_count', 'age'])
    >>> print(matches)
    {'days_since_last_purchase': 'recency', 'order_count': 'frequency', 'age': None}
    """

    def __init__(
        self,
        pretrained_features: List[str],
        aliases_path: Optional[Path] = None
    ):
        # Normalize pre-trained features to lowercase for case-insensitive matching
        self.pretrained_features = [f.lower() for f in pretrained_features]

        # Load aliases
        if aliases_path is None:
            aliases_path = Path(__file__).parent / "data" / "feature_aliases.json"

        self.aliases = self._load_aliases(aliases_path)
        self.reverse_alias_map = self._build_reverse_alias_map()

    def _load_aliases(self, aliases_path: Path) -> Dict[str, List[str]]:
        """Load feature aliases from JSON file."""
        if not aliases_path.exists():
            return {}

        with open(aliases_path, 'r') as f:
            aliases = json.load(f)

        # Normalize all keys and values to lowercase
        normalized_aliases = {}
        for key, values in aliases.items():
            normalized_key = key.lower()
            normalized_values = [v.lower() for v in values]
            normalized_aliases[normalized_key] = normalized_values

        return normalized_aliases

    def _build_reverse_alias_map(self) -> Dict[str, str]:
        """Build reverse mapping from alias -> pre-trained feature."""
        reverse_map = {}

        for pretrained_feature, alias_list in self.aliases.items():
            for alias in alias_list:
                # Store mapping from alias to pre-trained feature
                reverse_map[alias] = pretrained_feature

        return reverse_map

    def match(self, user_feature: str) -> Optional[Tuple[str, float]]:
        """
        Match a single user feature to a pre-trained feature.

        Parameters
        ----------
        user_feature : str
            User's feature name

        Returns
        -------
        result : Optional[Tuple[str, float]]
            Tuple of (matched_pretrained_feature, confidence_score)
            Returns None if no match found
            Confidence is always 1.0 for Tier 1 and Tier 2 matches
        """
        user_feature_normalized = user_feature.lower()

        # Tier 1: Exact match (case-insensitive)
        if user_feature_normalized in self.pretrained_features:
            return (user_feature_normalized, 1.0)

        # Tier 2: Alias match
        if user_feature_normalized in self.reverse_alias_map:
            matched_feature = self.reverse_alias_map[user_feature_normalized]
            # Verify the matched feature is actually in pre-trained features
            if matched_feature in self.pretrained_features:
                return (matched_feature, 1.0)

        # No match found
        return None

    def match_all(
        self,
        user_features: List[str]
    ) -> Tuple[Dict[str, Optional[str]], List[Dict]]:
        """
        Match all user features to pre-trained features.

        Parameters
        ----------
        user_features : List[str]
            List of user's feature names

        Returns
        -------
        matches : Dict[str, Optional[str]]
            Dictionary mapping user_feature -> matched_pretrained_feature
            Value is None if no match found

        match_info : List[Dict]
            Detailed information about each match, including:
            - user_feature: Original user feature name
            - matched_to: Matched pre-trained feature (or None)
            - confidence: Match confidence (0.0 to 1.0)
            - match_type: 'exact', 'alias', or 'no_match'
        """
        matches = {}
        match_info = []

        for uf in user_features:
            result = self.match(uf)

            if result:
                matched_feature, confidence = result
                matches[uf] = matched_feature

                # Determine match type
                if uf.lower() == matched_feature:
                    match_type = 'exact'
                else:
                    match_type = 'alias'

                match_info.append({
                    'user_feature': uf,
                    'matched_to': matched_feature,
                    'confidence': confidence,
                    'match_type': match_type
                })
            else:
                matches[uf] = None
                match_info.append({
                    'user_feature': uf,
                    'matched_to': None,
                    'confidence': 0.0,
                    'match_type': 'no_match'
                })

        return matches, match_info

    def get_match_statistics(
        self,
        user_features: List[str]
    ) -> Dict:
        """
        Get summary statistics about feature matching.

        Parameters
        ----------
        user_features : List[str]
            List of user's feature names

        Returns
        -------
        stats : Dict
            Dictionary with matching statistics:
            - total_features: Total number of user features
            - matched_features: Number of matched features
            - unmatched_features: Number of unmatched features
            - match_rate: Proportion of matched features (0.0 to 1.0)
            - exact_matches: Number of exact matches
            - alias_matches: Number of alias matches
        """
        _, match_info = self.match_all(user_features)

        matched = [m for m in match_info if m['matched_to'] is not None]
        exact = [m for m in match_info if m['match_type'] == 'exact']
        alias = [m for m in match_info if m['match_type'] == 'alias']

        total = len(user_features)
        matched_count = len(matched)

        return {
            'total_features': total,
            'matched_features': matched_count,
            'unmatched_features': total - matched_count,
            'match_rate': matched_count / total if total > 0 else 0.0,
            'exact_matches': len(exact),
            'alias_matches': len(alias)
        }

    def print_match_report(
        self,
        user_features: List[str],
        verbose: bool = True
    ) -> None:
        """
        Print a detailed report of feature matching results.

        Parameters
        ----------
        user_features : List[str]
            List of user's feature names
        verbose : bool, optional (default=True)
            If True, prints detailed match information for each feature
        """
        matches, match_info = self.match_all(user_features)
        stats = self.get_match_statistics(user_features)

        print("\n" + "=" * 70)
        print("Feature Matching Results")
        print("=" * 70)
        print(f"\nAnalyzing {stats['total_features']} user features "
              f"against {len(self.pretrained_features)} pre-trained features...\n")

        # Matched features
        matched_info = [m for m in match_info if m['matched_to'] is not None]
        if matched_info:
            print(f"✓ Matched features ({len(matched_info)}):")
            if verbose:
                for m in matched_info:
                    if m['match_type'] == 'exact':
                        print(f"  • '{m['user_feature']}' → '{m['matched_to']}' "
                              f"(exact match, confidence: {m['confidence']:.2f})")
                    else:
                        print(f"  • '{m['user_feature']}' → '{m['matched_to']}' "
                              f"(alias match, confidence: {m['confidence']:.2f})")
            print()

        # Unmatched features
        unmatched_info = [m for m in match_info if m['matched_to'] is None]
        if unmatched_info:
            print(f"⚠ Unmatched features ({len(unmatched_info)}):")
            if verbose:
                for m in unmatched_info:
                    print(f"  • '{m['user_feature']}' (no semantic match found)")
            print()

        # Summary
        match_rate = stats['match_rate']
        print(f"Transfer Learning Coverage: {match_rate:.1%} "
              f"({stats['matched_features']}/{stats['total_features']} features)")
        print("→ Using pre-trained priors for matched features")
        print("→ Using weakly informative priors for unmatched features")

        # Fixed tau (no longer adaptive to avoid convergence issues)
        print(f"\nBetween-SME variance (tau): 2.0 (standard)")
        print("=" * 70 + "\n")
