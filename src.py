if self._cfg.phrase_sets:
            from google.cloud.speech_v2.types import cloud_speech as cs
            
            # ── PATCH: Handle undocumented API combinations and limits ──
            if self._cfg.model.startswith("telephony"):
                self.log.warning(
                    "[%s] Model '%s' does not support V2 phrase sets. "
                    "Stripping adaptation entirely to prevent API crash.",
                    self.engine_id, self._cfg.model
                )
            elif self._cfg.model == "chirp_3" and self._cfg.enable_diarization:
                self.log.warning(
                    "[%s] 'chirp_3' cannot combine Diarization and Phrase Sets simultaneously (causes 404). "
                    "Stripping adaptation to allow Diarization to succeed.",
                    self.engine_id
                )
            else:
                adaptation_phrase_sets = []
                for ps in self._cfg.phrase_sets:
                    phrases = []
                    for p in ps.get("phrases", []):
                        if isinstance(p, dict):
                            phrases.append(
                                cs.PhraseSet.Phrase(
                                    value=p.get("value", ""), 
                                    boost=p.get("boost", 0.0)
                                )
                            )
                        else:
                            phrases.append(cs.PhraseSet.Phrase(value=str(p)))
                            
                    adaptation_phrase_sets.append(
                        cs.SpeechAdaptation.AdaptationPhraseSet(
                            inline_phrase_set=cs.PhraseSet(
                                phrases=phrases,
                                boost=ps.get("boost", 0.0)
                            )
                        )
                    )
                kwargs["adaptation"] = cs.SpeechAdaptation(phrase_sets=adaptation_phrase_sets)
