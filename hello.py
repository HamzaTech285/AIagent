
# for chainlit
import chainlit as cl

# for api key
import os
from dotenv import load_dotenv, find_dotenv

# for openAI agent SDK
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

load_dotenv(find_dotenv())

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

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True,
)


# 4 Agent

agent = Agent(
    instructions="Act as a helpful assistant.",
    name="Gemini Assistant",
)





# Starting Masssage and History
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! How can I assist you today?").send()# starting massage






# For Chainlit User Interface
@cl.on_message
async def handle_massage(message: cl.Message):
    # History
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # Runner
    result = await Runner.run(
        agent,
        input=message.content,
        run_config=run_config,
        )
    
    #History
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    # Output / llm Respond
    await cl.Message(content=result.final_output).send()



