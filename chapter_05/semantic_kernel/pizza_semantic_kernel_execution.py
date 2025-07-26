import asyncio

import httpx
import semantic_kernel as sk
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions.kernel_function_decorator import kernel_function

from readkey import get_token

selected_service = "OpenAI"

api_key = str(get_token())
service_id = None
custom_async_http_client = httpx.AsyncClient(verify=False)
openai_client_with_custom_http = AsyncOpenAI(
            api_key=api_key,
            http_client=custom_async_http_client # Pass the custom httpx.AsyncClient here
        )

class PizzaPlugin:
    @kernel_function(description="Checks balance amount in rupees on users pizza wallet; returns the balance amount")
    def get_pizza_wallet_balance(self, wallet_password:str):
        # may be we can integrate a real wallet service here to get the balance amount
        print("Invoked get_pizza_wallet_balance function !!")
        balance = 144.34
        return f"balance : Rs.{balance}"

    @kernel_function(description="Checks for available pizzas and return them.")
    def get_available_pizza(self):
        # this is static data and later may be we can  setup a database kind of
        pizzas = {"Pizza 1" : {"Name" : "Bryon's Bigdamaka pizza", "Price" : 180.76},
        "Pizza 2" : {"Name" : "Gramin's Small Pizza", "Price" : 129.87},
        "Pizza 3" : {"Name" : "Jaorin's Special Pizza", "Price" : 239.76},
        }
        print(f"Invoked get_available_pizza function !!")
        return str(pizzas)

    @kernel_function(description="Order a pizza with the given pizza name and user wallet balance; return the confirmation message for the order")
    def order_pizza(self, pizza_name:str, pizza_price: float, wallet_balance:float):
        # her we can use some logic to detect the amount from user wallet
        print("Invoked Pizza order function !!")
        if wallet_balance < pizza_price:
            return f"You wallet balance is insufficient to place an order for {pizza_name}. Please recharge your wallet."
        return f"Your order for {pizza_name} has been placed successfully."




async def run_prompt():
    kernel = sk.Kernel()
    kernel.add_plugin(PizzaPlugin(), plugin_name="OrderPizzaPlugin")
    service_id = "oai_chat_gpt"
    chat_completion_service =OpenAIChatCompletion(
            service_id=service_id,
            ai_model_id="gpt-4o-mini",
            api_key=api_key,
            org_id="org-MAd9CASqY83u8gtOXsyQP5Lx",
            async_client=openai_client_with_custom_http
        )
    kernel.add_service(chat_completion_service)
    chat_history = ChatHistory()
    chat_history.add_system_message("Your name is 'Pizzer' and you are a pizza ordering agent. You can order pizza for the user. You can also check the available pizzas at the moment, and additionally each users has the pizza wallet to order, you can check the balance amount in the wallet.")


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

