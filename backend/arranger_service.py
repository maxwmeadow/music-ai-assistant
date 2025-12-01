"""
AI Arranger Service
Handles LLM-based music arrangement generation using Claude API
"""

import os
import re
from typing import Dict, Optional
import anthropic
from anthropic import Anthropic

from backend.llm_prompts import DSL_DOCUMENTATION, build_arrangement_prompt
from backend.dsl_context import extract_tempo, build_music_context
from backend.instrument_descriptions import (
    get_instruments_for_track_type,
    format_instruments_for_prompt
)


class ArrangerService:
    """Service for generating music arrangements using Claude API."""

    def __init__(self, api_key: Optional[str] = None, default_temperature: float = 0.8):
        """
        Initialize the arranger service.

        Args:
            api_key: Optional Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var
            default_temperature: Default temperature for generation (0.0-1.0, default 0.8)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key to constructor."
            )

        self.client = Anthropic(api_key=self.api_key)

        # Model configuration
        self.model = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5
        self.default_temperature = max(0.0, min(1.0, default_temperature))  # Clamp to 0-1
        self.max_tokens = 4096

    def generate_arrangement(
        self,
        current_dsl: str,
        track_type: str,
        genre: str,
        custom_request: Optional[str] = None,
        include_key: bool = True,
        include_chords: bool = True,
        creativity: float = 0.7,
        complexity: str = "medium"
    ) -> str:
        """
        Generate a complementary track using Claude API.

        Args:
            current_dsl: The existing DSL code
            track_type: Type of track to generate (bass, chords, pad, melody, counterMelody, arpeggio, drums)
            genre: Musical genre/style
            custom_request: Optional additional instructions from user
            include_key: Whether to attempt key detection
            include_chords: Whether to extract chord progression
            creativity: Creativity level 0.0-1.0 (0=safe/predictable, 1=experimental/varied)
            complexity: Complexity level - "simple", "medium", or "complex"

        Returns:
            Generated DSL code for the new track

        Raises:
            Exception: If API call fails or response is invalid
        """
        # Extract musical context from existing DSL
        tempo = extract_tempo(current_dsl)
        music_context = build_music_context(
            current_dsl,
            include_key=include_key,
            include_chords=include_chords
        )

        # Get instruments for this track type
        instrument_paths = get_instruments_for_track_type(track_type)
        if not instrument_paths:
            raise ValueError(f"No instruments found for track type: {track_type}")

        # Format instruments with descriptions
        available_instruments = format_instruments_for_prompt(instrument_paths)

        # Map creativity to temperature (0.0-1.0 → 0.5-1.0)
        # Lower bound of 0.5 ensures some variation, upper bound of 1.0 allows max creativity
        temperature = 0.5 + (creativity * 0.5)
        temperature = max(0.5, min(1.0, temperature))  # Clamp to safe range

        # Build the complete prompt
        user_message = build_arrangement_prompt(
            current_dsl=current_dsl,
            track_type=track_type,
            genre=genre,
            available_instruments=available_instruments,
            tempo=tempo,
            user_request=custom_request,
            music_context=music_context,
            complexity=complexity
        )

        # Call Claude API with prompt caching
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,  # Use calculated temperature
                system=[
                    {
                        "type": "text",
                        "text": DSL_DOCUMENTATION,
                        "cache_control": {"type": "ephemeral"}  # Cache the DSL documentation
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": user_message
                    }
                ]
            )

            # Extract generated DSL from response
            if not response.content:
                raise Exception("Empty response from Claude API")

            # Get the text content from the response
            generated_dsl = response.content[0].text

            # Clean up the response (remove any markdown fences if present)
            generated_dsl = self._clean_response(generated_dsl)

            # Validate the generated DSL
            is_valid, validation_errors = self._validate_dsl(generated_dsl)

            if not is_valid:
                error_msg = "Generated DSL failed validation:\n" + "\n".join(f"  - {err}" for err in validation_errors)
                print(f"[VALIDATION ERROR] {error_msg}")
                print(f"[VALIDATION ERROR] Generated code:\n{generated_dsl}")
                raise Exception(f"Validation failed: {'; '.join(validation_errors)}")

            print("[VALIDATION] Generated DSL passed all validation checks")
            return generated_dsl

        except anthropic.APIError as e:
            raise Exception(f"Claude API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to generate arrangement: {str(e)}")

    def _clean_response(self, text: str) -> str:
        """
        Clean up the LLM response to extract pure DSL code.

        Args:
            text: Raw response text

        Returns:
            Cleaned DSL code
        """
        # Remove markdown code fences if present
        text = text.strip()

        # Remove opening code fence
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```dsl or ```javascript or just ```)
            lines = lines[1:]
            text = "\n".join(lines)

        # Remove closing code fence
        if text.endswith("```"):
            text = text[:-3].rstrip()

        # Remove any explanatory text before the track() command
        # Find the first occurrence of track(
        track_start = text.find("track(")
        if track_start > 0:
            text = text[track_start:]

        return text.strip()

    def _validate_dsl(self, dsl_code: str) -> tuple[bool, list[str]]:
        """
        Validate generated DSL code for common errors.

        Args:
            dsl_code: The DSL code to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if code is not empty
        if not dsl_code or not dsl_code.strip():
            errors.append("Generated code is empty")
            return False, errors

        # Check if starts with track(
        if not dsl_code.strip().startswith("track("):
            errors.append("Code must start with track()")

        # Check if ends with }
        if not dsl_code.strip().endswith("}"):
            errors.append("Code must end with }")

        # Check for instrument() command
        if "instrument(" not in dsl_code:
            errors.append("Missing instrument() command")

        # Note: We no longer validate flat vs sharp notes since the runner
        # automatically converts flats to sharps (Bb→A#, Eb→D#, etc.)

        # Check for velocity out of range (0.0-1.0)
        # Pattern: note("...", ..., ..., velocity) or chord([...], ..., ..., velocity)
        velocity_pattern = r'(?:note|chord)\([^)]+,\s*([\d.]+)\s*\)'
        velocities = re.findall(velocity_pattern, dsl_code)

        invalid_velocities = []
        for vel in velocities:
            try:
                vel_float = float(vel)
                if vel_float < 0.0 or vel_float > 1.0:
                    invalid_velocities.append(vel)
            except ValueError:
                invalid_velocities.append(vel)

        if invalid_velocities:
            errors.append(f"Velocity must be 0.0-1.0, found: {', '.join(invalid_velocities)}")

        # Check for negative start times
        # Pattern: note("...", start, ..., ...) or chord([...], start, ..., ...)
        start_time_pattern = r'(?:note|chord)\([^,]+,\s*([-\d.]+)\s*,'
        start_times = re.findall(start_time_pattern, dsl_code)

        negative_times = []
        for time in start_times:
            try:
                time_float = float(time)
                if time_float < 0:
                    negative_times.append(time)
            except ValueError:
                pass

        if negative_times:
            errors.append(f"Start time cannot be negative, found: {', '.join(negative_times)}")

        # Check for matching braces and parentheses
        open_braces = dsl_code.count('{')
        close_braces = dsl_code.count('}')
        if open_braces != close_braces:
            errors.append(f"Mismatched braces: {open_braces} {{ vs {close_braces} }}")

        open_parens = dsl_code.count('(')
        close_parens = dsl_code.count(')')
        if open_parens != close_parens:
            errors.append(f"Mismatched parentheses: {open_parens} ( vs {close_parens} )")

        # Check for tempo() command (should not be present)
        if "tempo(" in dsl_code:
            errors.append("Generated code should not include tempo() - already set")

        is_valid = len(errors) == 0
        return is_valid, errors


# Singleton instance
_arranger_service: Optional[ArrangerService] = None


def get_arranger_service() -> ArrangerService:
    """Get or create the singleton arranger service instance."""
    global _arranger_service

    if _arranger_service is None:
        _arranger_service = ArrangerService()

    return _arranger_service