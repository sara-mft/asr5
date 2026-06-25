if self._cfg.enable_diarization:
            # ── PATCH: Prevent diarization crash on unsupported models ──
            if self._cfg.model.startswith("telephony") or self._cfg.model == "chirp_2":
                self.log.warning(
                    "[%s] Model '%s' does not support speaker_diarization in the V2 API. "
                    "Stripping diarization_config entirely to prevent API crash.",
                    self.engine_id, self._cfg.model
                )
            else:
                diar_kwargs: dict[str, Any] = {}
                if self._cfg.min_speaker_count:
                    diar_kwargs["min_speaker_count"] = self._cfg.min_speaker_count
                if self._cfg.max_speaker_count:
                    diar_kwargs["max_speaker_count"] = self._cfg.max_speaker_count
                
                features_kwargs["diarization_config"] = cs.SpeakerDiarizationConfig(**diar_kwargs)
