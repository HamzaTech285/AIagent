from typing import cast
# for chainlit
import chainlit as cl

# for api key
import os
from dotenv import load_dotenv, find_dotenv

# for openAI agent SDK
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

load_dotenv(find_dotenv())

# API KEY
gemini_api_key = os.getenv("GEMINI_API_KEY")



# 1 Provider

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# 2 Model

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider,
)

# 3 Config

config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)


# 4 Agent

agent = Agent(
    instructions="Act as a helpful assistant.",
    name="Gemini Assistant",
)


# ///////////////////////////////
# CODE FOR STREAMING IN CHAINLIT        STREAMING  #   5
# ///////////////////////////////


# Starting Masssage and History
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])

    cl.user_session.set("config", config)
    
    cl.user_session.set("agent", agent)
    await cl.Message(content="Hello! How can I assist you today?").send()# starting massage




# For Chainlit User Interface
@cl.on_message
async def handle_massage(message: cl.Message):
    # History
    history = cl.user_session.get("history") or []
    history.append({"role": "user", "content": message.content})

    # Create a new message object for streaming
    msg = cl.Message(content="")
    await msg.send()

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        # Run the agent with streaming enabled
        result = Runner.run_streamed(agent, history, run_config=config)

        # Stream the response token by token
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

        # Append the assistant's response to the history.
        history.append({"role": "assistant", "content": msg.content})

        # Update the session with the new history.
        cl.user_session.set("chat_history", history)

        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {msg.content}")

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"Error: {str(e)}")






