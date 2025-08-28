import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    ModelSettings,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice.transcription.filters import filter_markdown
from livekit.plugins import deepgram, openai, silero, cartesia, google
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor.",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        self.session.generate_reply()

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        # TTS node allows us to process the text before it's sent to the model
        # here we'll strip out markdown
        filtered_text = filter_markdown(text)
        return super().tts_node(filtered_text, model_settings)

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, please estimate the latitude and longitude of the location and
        do not ask the user for them.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location, do not ask user for it
            longitude: The longitude of the location, do not ask user for it
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # each log entry will include these fields
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=google.LLM(model="gemini-2.0-flash",temperature=0.8),
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=cartesia.TTS(model="sonic-2",voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        # allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
        # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
        # when it's detected, you may resume the agent's speech
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        min_interruption_duration=0.2,  # with false interruption resume, interruption can be more sensitive
        # use LiveKit's turn detection model
        # turn_detection=MultilingualModel(),
    )

    # log metrics as they are emitted, and total usage after session is over
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    # shutdown callbacks are triggered when the session is over
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

# import logging
# from collections.abc import AsyncIterable

# from dotenv import load_dotenv

# from livekit import rtc
# from livekit.agents import (
#     NOT_GIVEN,
#     Agent,
#     AgentFalseInterruptionEvent,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     MetricsCollectedEvent,
#     ModelSettings,
#     RoomInputOptions,
#     RoomOutputOptions,
#     RunContext,
#     WorkerOptions,
#     cli,
#     metrics,
# )
# from livekit.agents.llm import function_tool
# from livekit.plugins import google
# from google.genai import types
# from livekit.agents.voice.transcription.filters import filter_markdown
# from livekit.plugins import deepgram, silero, cartesia
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# # uncomment to enable Krisp background voice/noise cancellation
# # from livekit.plugins import noise_cancellation

# logger = logging.getLogger("basic-agent")

# load_dotenv()


# class MyAgent(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="Your name is Kelly. You would interact with users via voice."
#             "with that in mind keep your responses concise and to the point."
#             "do not use emojis, asterisks, markdown, or other special characters in your responses."
#             "You are curious and friendly, and have a sense of humor.",
#         )

#     async def on_enter(self):
#         # when the agent is added to the session, it'll generate a reply
#         # according to its instructions
#         self.session.generate_reply()

#     async def tts_node(
#         self, text: AsyncIterable[str], model_settings: ModelSettings
#     ) -> AsyncIterable[rtc.AudioFrame]:
#         # TTS node allows us to process the text before it's sent to the model
#         # here we'll strip out markdown
#         filtered_text = filter_markdown(text)
#         return super().tts_node(filtered_text, model_settings)

#     # all functions annotated with @function_tool will be passed to the LLM when this
#     # agent is active
#     @function_tool
#     async def lookup_weather(
#         self, context: RunContext, location: str, latitude: str, longitude: str
#     ):
#         """Called when the user asks for weather related information.
#         Ensure the user's location (city or region) is provided.
#         When given a location, please estimate the latitude and longitude of the location and
#         do not ask the user for them.

#         Args:
#             location: The location they are asking for
#             latitude: The latitude of the location, do not ask user for it
#             longitude: The longitude of the location, do not ask user for it
#         """

#         logger.info(f"Looking up weather for {location}")

#         return "sunny with a temperature of 70 degrees."


# def prewarm(proc: JobProcess):
#     proc.userdata["vad"] = silero.VAD.load()


# async def entrypoint(ctx: JobContext):
#     # each log entry will include these fields
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }

#     session = AgentSession(
#         vad=ctx.proc.userdata["vad"],
#         # any combination of STT, LLM, TTS, or realtime API can be used
#         llm=google.LLM(
#         model="gemini-2.0-flash-exp",
#         gemini_tools=[types.GoogleSearch()],
#          ),
#         stt=deepgram.STT(model="nova-3", language="multi"),
#         tts=cartesia.TTS(
#         model="sonic-2",
#         voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
#           ),
#     )

#     # log metrics as they are emitted, and total usage after session is over
#     usage_collector = metrics.UsageCollector()

#     # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
#     # when it's detected, you may resume the agent's speech
#     @session.on("agent_false_interruption")
#     def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
#         logger.info("false positive interruption, resuming")
#         session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

#     @session.on("metrics_collected")
#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         metrics.log_metrics(ev.metrics)
#         usage_collector.collect(ev.metrics)

#     async def log_usage():
#         summary = usage_collector.get_summary()
#         logger.info(f"Usage: {summary}")

#     # shutdown callbacks are triggered when the session is over
#     ctx.add_shutdown_callback(log_usage)

#     await session.start(
#         agent=MyAgent(),
#         room=ctx.room,
#         room_input_options=RoomInputOptions(
#             # uncomment to enable Krisp BVC noise cancellation
#             # noise_cancellation=noise_cancellation.BVC(),
#         ),
#         room_output_options=RoomOutputOptions(transcription_enabled=True),
#     )


# if __name__ == "__main__":
#     cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


# import logging
# from collections.abc import AsyncIterable

# from dotenv import load_dotenv

# from livekit import rtc
# from livekit.agents import (
#     Agent,
#     AgentSession,
#     JobContext,
#     JobProcess,
#     MetricsCollectedEvent,
#     ModelSettings,
#     RoomInputOptions,
#     RoomOutputOptions,
#     RunContext,
#     WorkerOptions,
#     cli,
#     metrics,
# )
# from livekit.agents.llm import function_tool
# from livekit.agents.voice.transcription.filters import filter_markdown
# from livekit.plugins import deepgram, openai, silero, cartesia, google

# # Remove the problematic multilingual turn detector import for now
# # from livekit.plugins.turn_detector.multilingual import MultilingualModel

# logger = logging.getLogger("basic-agent")

# load_dotenv()


# class MyAgent(Agent):
#     def __init__(self) -> None:
#         super().__init__(
#             instructions="Your name is Kelly. You would interact with users via voice."
#             "with that in mind keep your responses concise and to the point."
#             "do not use emojis, asterisks, markdown, or other special characters in your responses."
#             "You are curious and friendly, and have a sense of humor.",
#         )

#     async def on_enter(self):
#         # when the agent is added to the session, it'll generate a reply
#         # according to its instructions
#         self.session.generate_reply()

#     async def tts_node(
#         self, text: AsyncIterable[str], model_settings: ModelSettings
#     ) -> AsyncIterable[rtc.AudioFrame]:
#         # TTS node allows us to process the text before it's sent to the model
#         # here we'll strip out markdown
#         filtered_text = filter_markdown(text)
#         return super().tts_node(filtered_text, model_settings)

#     @function_tool
#     async def lookup_weather(
#         self, context: RunContext, location: str, latitude: str, longitude: str
#     ):
#         """Called when the user asks for weather related information.
#         Ensure the user's location (city or region) is provided.
#         When given a location, please estimate the latitude and longitude of the location and
#         do not ask the user for them.

#         Args:
#             location: The location they are asking for
#             latitude: The latitude of the location, do not ask user for it
#             longitude: The longitude of the location, do not ask user for it
#         """

#         logger.info(f"Looking up weather for {location}")

#         return "sunny with a temperature of 70 degrees."


# def prewarm(proc: JobProcess):
#     try:
#         # Add timeout and error handling for VAD loading
#         proc.userdata["vad"] = silero.VAD.load()
#         logger.info("VAD model loaded successfully")
#     except Exception as e:
#         logger.error(f"Failed to load VAD model: {e}")
#         # Use None as fallback - agent will work without VAD
#         proc.userdata["vad"] = None


# async def entrypoint(ctx: JobContext):
#     # each log entry will include these fields
#     ctx.log_context_fields = {
#         "room": ctx.room.name,
#     }

#     # Get VAD from prewarm, with fallback
#     vad_model = ctx.proc.userdata.get("vad")
#     if vad_model is None:
#         logger.warning("VAD model not available, using voice activity detection from STT")

#     session = AgentSession(
#         vad=vad_model,  # This can be None and agent will still work
#         llm=google.LLM(model="gemini-2.0-flash", temperature=0.8),
#         stt=deepgram.STT(model="nova-2", language="en"),  # Use nova-2 instead of nova-3, single language
#         tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        
#         # Simplified settings to reduce initialization complexity
#         preemptive_generation=False,  # Disable for now
#         resume_false_interruption=False,  # Disable for now
#         min_interruption_duration=0.5,  # More conservative
        
#         # Remove the problematic multilingual turn detection
#         # turn_detection=MultilingualModel(),  # Comment this out
#     )

#     # log metrics as they are emitted, and total usage after session is over
#     usage_collector = metrics.UsageCollector()

#     @session.on("metrics_collected")
#     def _on_metrics_collected(ev: MetricsCollectedEvent):
#         metrics.log_metrics(ev.metrics)
#         usage_collector.collect(ev.metrics)

#     async def log_usage():
#         summary = usage_collector.get_summary()
#         logger.info(f"Usage: {summary}")

#     ctx.add_shutdown_callback(log_usage)

#     try:
#         await session.start(
#             agent=MyAgent(),
#             room=ctx.room,
#             room_input_options=RoomInputOptions(),
#             room_output_options=RoomOutputOptions(transcription_enabled=True),
#         )
#     except Exception as e:
#         logger.error(f"Failed to start session: {e}")
#         raise


# if __name__ == "__main__":
#     # Add logging configuration
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
    
#     cli.run_app(WorkerOptions(
#         entrypoint_fnc=entrypoint, 
#         prewarm_fnc=prewarm,
#     ))