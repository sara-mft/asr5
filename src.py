if self._cfg.denoise_audio:
            from google.cloud.speech_v2.types import cloud_speech as cs
            
            # ── PATCH: Prevent denoiser crash on telephony models ──
            if self._cfg.model.startswith("telephony"):
                self.log.warning(
                    "[%s] Model '%s' does not support the denoiser feature. "
                    "Stripping denoiser_config entirely to prevent API crash.",
                    self.engine_id, self._cfg.model
                )
            else:
                kwargs["denoiser_config"] = cs.DenoiserConfig(
                    denoise_audio=True, snr_threshold=self._cfg.snr_threshold
                )
