import asyncio

import semantic_kernel as sk
from readkey import get_token

selected_service = "OpenAI"
kernel = sk.Kernel()
api_key = str(get_token())
service_id = None
if selected_service == "OpenAI":
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    service_id = "oai_chat_gpt"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-4o-mini",
            api_key=api_key,
            org_id="org-<>",
        ),
    )



# This function is currently broken
async def run_prompt():
    result = await kernel.invoke_prompt(prompt="recommend a movie about time travel")
    print(result)


# Use asyncio.run to execute the async function
asyncio.run(run_prompt())
