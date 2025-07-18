import asyncio
import json
import semantic_kernel as sk
from readkey import get_token
import httpx
from openai import AsyncOpenAI
from semantic_kernel.functions.kernel_function_decorator import kernel_function

selected_service = "OpenAI"
kernel = sk.Kernel()
api_key = str(get_token())
service_id = None
custom_async_http_client = httpx.AsyncClient(verify=False)
openai_client_with_custom_http = AsyncOpenAI(
            api_key=api_key,
            http_client=custom_async_http_client # Pass the custom httpx.AsyncClient here
        )
@kernel_function(name="recommend_fun", description="Get the weather for a city")
def recommend_fun(topic, rating="good"):
    if "time travel" in topic.lower():     #1
        return json.dumps({"topic": "time travel",
                           "recommendation": "Back to the Future",
                           "rating": rating})
    elif "recipe" in topic.lower():    #1
        return json.dumps({"topic": "recipe",
                           "recommendation": "The best thing … ate.",
                           "rating": rating})
    elif "gift" in topic.lower():      #1
        return json.dumps({"topic": "gift",
                           "recommendation": "A glorious new...",
                           "rating": rating})
    else:     #2
        return json.dumps({"topic": topic,
                           "recommendation": "unknown"})

if selected_service == "OpenAI":
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

    service_id = "oai_chat_gpt"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-4o-mini",
            api_key=api_key,
            org_id="org-MAd9CASqY83u8gtOXsyQP5Lx",
            async_client=openai_client_with_custom_http
        ),
    )



async def run_prompt():
    result = await kernel.invoke_prompt(prompt="recommend a movie about time travel",
                                        function_name="recommend_fun")
    print("Result for sematic kernel is ::::")
    print(result)


# Use asyncio.run to execute the async function
asyncio.run(run_prompt())
