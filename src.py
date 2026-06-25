if self._cfg.phrase_sets:
            from google.cloud.speech_v2.types import cloud_speech as cs
            
            # The V2 adaptation field is strictly unsupported by telephony models.
            if self._cfg.model.startswith("telephony"):
                self.log.warning(
                    "[%s] Model '%s' does not support V2 phrase sets. "
                    "Stripping adaptation entirely to prevent API crash. "
                    "(Requires V1 API 'speech_contexts' for phrase boosting).",
                    self.engine_id, self._cfg.model
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
                # Only attach the adaptation block for models that actually support V2 adaptation
                kwargs["adaptation"] = cs.SpeechAdaptation(phrase_sets=adaptation_phrase_sets)
