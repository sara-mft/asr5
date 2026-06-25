if self._cfg.phrase_sets:
            from google.cloud.speech_v2.types import cloud_speech as cs
            
            adaptation_phrase_sets = []
            # Flag if the chosen model supports the 'boost' feature
            supports_boost = not self._cfg.model.startswith("telephony")

            for ps in self._cfg.phrase_sets:
                safe_phrases = []
                
                # 1. Sanitize phrase-level boosts
                for p in ps.get("phrases", []):
                    if isinstance(p, dict):
                        p_kwargs = {"value": p.get("value", "")}
                        if supports_boost and "boost" in p:
                            p_kwargs["boost"] = p["boost"]
                        safe_phrases.append(cs.PhraseSet.Phrase(**p_kwargs))
                    else:
                        # Fallback if config just provides a list of strings
                        safe_phrases.append(cs.PhraseSet.Phrase(value=str(p)))

                phrase_set_kwargs = {"phrases": safe_phrases}
                
                # 2. Sanitize PhraseSet-level boost
                if supports_boost and "boost" in ps:
                    phrase_set_kwargs["boost"] = ps["boost"]

                # 3. Log a warning so you know the weights were dropped
                if not supports_boost and ("boost" in ps or any(isinstance(p, dict) and "boost" in p for p in ps.get("phrases", []))):
                    self.log.warning(
                        "[%s] Model '%s' does not support speech_adaptation_boost. "
                        "Stripping boost weights to prevent API crash.",
                        self.engine_id, self._cfg.model
                    )

                adaptation_phrase_sets.append(
                    cs.SpeechAdaptation.AdaptationPhraseSet(
                        inline_phrase_set=cs.PhraseSet(**phrase_set_kwargs)
                    )
                )

            kwargs["adaptation"] = cs.SpeechAdaptation(phrase_sets=adaptation_phrase_sets)
