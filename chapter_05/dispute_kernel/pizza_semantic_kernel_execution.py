import asyncio

import httpx
import semantic_kernel as sk
import requests
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from readkey import get_token,get_org_id

selected_service = "OpenAI"

api_key = str(get_token())
org_id = str(get_org_id())
service_id = None
custom_async_http_client = httpx.AsyncClient(verify=False)
openai_client_with_custom_http = AsyncOpenAI(
    api_key=api_key,
    http_client=custom_async_http_client  # Pass the custom httpx.AsyncClient here
)

auth_token= "O-Bearer <>"

def call_dispute_service_date_range(start_date,end_date):
    try:
        print("Calling the date range from start_date and end_date :: "+ str(start_date) + " :: "+ str(end_date))
        url = "http://localhost:8080/v1/chargeback/states"
        HEADERS = { 'accept': 'application/json',
                'Content-Type': 'application/json'}
        HEADERS['Authorization'] = auth_token
        response = requests.get(url=url,headers=HEADERS,verify=False)
        print(response)
    except Exception as e:
        print(e)

class DisputePlugin:
    # @kernel_function(description="Checks balance amount in rupees on users pizza wallet; returns the balance amount")
    @kernel_function(description="Get the list of disputes for use case date range filter")
    def get_dispute_data_range(self, start_date: str, end_date: str):
        # may be we can integrate a real wallet service here to get the balance amount
        print("Invoked date range filter for use case with date values : !!" + str(start_date) + " :: "+ str(end_date))
        call_dispute_service_date_range(start_date,end_date)
        balance = 144.34
        return f"balance : Rs.{balance}"



async def run_prompt():
    kernel = sk.Kernel()
    kernel.add_plugin(DisputePlugin(), plugin_name="DisputeManagementPlugin")
    service_id = "oai_chat_gpt"
    chat_completion_service = OpenAIChatCompletion(
        service_id=service_id,
        ai_model_id="gpt-4o-mini",
        api_key=api_key,
        org_id=org_id,
        async_client=openai_client_with_custom_http
    )
    kernel.add_service(chat_completion_service)
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "Your name is 'DisputeResolver' and you are a dispute management  agent. You can check the list of dispute on date range for the user.")

    execution_settings = OpenAIChatPromptExecutionSettings()
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    while True:
        user_input = input("Enter your message >>> ")
        if user_input.lower() == "q":
            print("You pressed 'q' exiting the program")
            break
        chat_history.add_user_message(user_input)
        response = await chat_completion_service.get_chat_message_content(
            chat_history=chat_history,
            settings=execution_settings,
            kernel=kernel,
        )
        response = str(response)
        chat_history.add_assistant_message(response)
        print("Response from agent >>> ", response)


asyncio.run(run_prompt())

# Use asyncio.run to execute the async function
