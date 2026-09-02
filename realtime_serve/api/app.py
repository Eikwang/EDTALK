"""FastAPI application for realtime streaming inference service."""
import base64
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from realtime_serve.audio_buffer import AudioBuffer


# Global state (will be initialized by main)
audio_buffer: AudioBuffer = None
render_loop = None
output_streamer = None
frame_buffer = None
character_data = None


class AudioPushRequest(BaseModel):
    """Request model for audio push endpoint (chunked streaming)."""
    pcm_base64: str
    timestamp_ms: int = 0
    seq: int = 0  # Sequence number for ordering


class AudioPushFullRequest(BaseModel):
    """Request model for full audio push endpoint.

    Pushes an entire audio segment at once. The server clears any
    existing buffer and loads the full segment, ensuring gapless
    playback without chunk-boundary artifacts.
    """
    pcm_base64: str


class SegmentSwitchRequest(BaseModel):
    """Request model for segment switch endpoint."""
    segment: str


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(title="EDTalk Realtime Streaming Inference Service")

    @app.post("/audio/push")
    async def push_audio(request: AudioPushRequest):
        """Push audio chunk to the buffer.

        Args:
            request: Audio push request with base64-encoded PCM data

        Returns:
            Success response with buffer duration
        """
        global audio_buffer

        if audio_buffer is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        try:
            # Decode base64 PCM data
            pcm_bytes = base64.b64decode(request.pcm_base64)

            # Push to buffer with sequence number for ordering
            audio_buffer.push(pcm_bytes, seq=request.seq)

            # Return buffer status
            buffer_ms = audio_buffer.buffer_duration_ms()
            return {"ok": True, "buffer_ms": round(buffer_ms)}

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")

    @app.post("/audio/push_full")
    async def push_audio_full(request: AudioPushFullRequest):
        """Push an entire audio segment at once.

        The full audio PCM is decoded and immediately processed through
        Audio2Lip (full-sequence inference matching the demo pipeline),
        producing all lip coefficients at once. The render loop then
        serves these coefficients one per frame, ensuring smooth,
        continuous lip animation.

        Args:
            request: Audio push request with base64-encoded full PCM data

        Returns:
            Success response with lip coefficient count and duration
        """
        global audio_buffer, render_loop

        if audio_buffer is None or render_loop is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        try:
            # Decode base64 PCM data
            pcm_bytes = base64.b64decode(request.pcm_base64)

            # Convert raw PCM to float32
            pcm_float = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Run Audio2Lip chunked inference in a background thread.
            # Playback starts as soon as the first chunk is ready, so this
            # returns immediately instead of blocking the HTTP response
            # (and the event loop) for the whole TTS duration.
            worker = threading.Thread(
                target=render_loop.process_full_audio,
                args=(pcm_float,),
                daemon=True,
            )
            worker.start()

            duration_s = len(pcm_float) / 16000.0
            return {"ok": True, "duration_s": round(duration_s, 2),
                    "processing": "async"}

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {str(e)}")

    @app.post("/segment/switch")
    async def switch_segment(request: SegmentSwitchRequest):
        """Switch to a different segment within the current character.

        Args:
            request: Segment switch request

        Returns:
            Success response with new segment info
        """
        global render_loop, character_data

        if render_loop is None or character_data is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        try:
            # Switch segment
            render_loop.switch_segment(request.segment)

            # Get updated segment info
            current_segment = character_data.current_segment
            total_frames = character_data.total_frames
            resolution = list(character_data.resolution)

            return {
                "ok": True,
                "segment": current_segment,
                "frame_count": total_frames,
                "resolution": resolution
            }

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to switch segment: {str(e)}")

    @app.get("/status")
    async def get_status():
        """Get current service status.

        Returns:
            Status information including production/output fps, buffer depth,
            underrun count, preroll state, current segment, etc.
        """
        global audio_buffer, render_loop, output_streamer, frame_buffer, character_data

        status = {
            "state": "running",
            "fps": 0.0,  # deprecated alias for output_fps
            "production_fps": 0.0,
            "output_fps": 0.0,
            "buffer_depth": 0,
            "buffer_ms": 0.0,
            "underruns": 0,
            "preroll_state": "unknown",
            "buffer_full_warn": False,
            "current_segment": None,
            "frame_index": 0,
            "resolution": None,
            "available_segments": []
        }

        if render_loop is not None:
            status["production_fps"] = round(render_loop.actual_fps, 1)
            status["frame_index"] = render_loop.frame_index

        if output_streamer is not None:
            streamer_stats = output_streamer.stats()
            status["output_fps"] = streamer_stats.get('target_fps', 0)
            status["underruns"] = streamer_stats.get('underruns', 0)
            status["preroll_state"] = streamer_stats.get('preroll_state', 'unknown')

        # Deprecated alias kept for one version.
        status["fps"] = status["output_fps"]

        if frame_buffer is not None:
            fb_stats = frame_buffer.stats()
            status["buffer_depth"] = fb_stats.get('depth', 0)
            status["buffer_full_warn"] = fb_stats.get('depth', 0) >= fb_stats.get('capacity', 1)

        if character_data is not None:
            status["current_segment"] = character_data.current_segment
            status["resolution"] = list(character_data.resolution)
            status["available_segments"] = character_data.segments

        if audio_buffer is not None:
            status["buffer_ms"] = round(audio_buffer.buffer_duration_ms(), 1)

        return status

    return app