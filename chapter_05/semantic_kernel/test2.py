import asyncio
import os
from typing import Annotated
from readkey import get_token
# --- Ensure your API key is loaded and printed ---
api_key_test = str(get_token())
# --- Import httpx.AsyncClient for SSL context modification ---
import httpx
from openai import AsyncOpenAI  # You might need this if passing a full OpenAI client
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import kernel_function


# Load your OpenAI API key from an environment variable or .env file
# It's recommended to use environment variables for security.
# You can set it as: export OPENAI_API_KEY="your_api_key_here"
# from dotenv import load_dotenv
# load_dotenv()

# --- 1. Define your Native Plugin Class ---
class MathPlugin:
    """
    A plugin that provides basic mathematical operations.
    """

    @kernel_function(
        name="add",
        description="Adds two numbers together."
    )
    def add(
        self,
        number1: Annotated[float, "The first number to add"],
        number2: Annotated[float, "The second number to add"]
    ) -> Annotated[float, "The sum of the two numbers"]:
        """Adds two numbers."""
        print(f"[MathPlugin] Calling add: {number1} + {number2}")
        return number1 + number2

    @kernel_function(
        name="subtract",
        description="Subtracts the second number from the first number."
    )
    def subtract(
        self,
        number1: Annotated[float, "The number to subtract from"],
        number2: Annotated[float, "The number to subtract"]
    ) -> Annotated[float, "The difference between the two numbers"]:
        """Subtracts two numbers."""
        print(f"[MathPlugin] Calling subtract: {number1} - {number2}")
        return number1 - number2

    @kernel_function(
        name="multiply",
        description="Multiplies two numbers together."
    )
    def multiply(
        self,
        number1: Annotated[float, "The first number to multiply"],
        number2: Annotated[float, "The second number to multiply"]
    ) -> Annotated[float, "The product of the two numbers"]:
        """Multiplies two numbers."""
        print(f"[MathPlugin] Calling multiply: {number1} * {number2}")
        return number1 * number2

    @kernel_function(
        name="divide",
        description="Divides the first number by the second number. Returns an error if division by zero."
    )
    def divide(
        self,
        number1: Annotated[float, "The dividend"],
        number2: Annotated[float, "The divisor"]
    ) -> Annotated[float, "The quotient of the two numbers"]:
        """Divides two numbers."""
        if number2 == 0:
            print("[MathPlugin] Error: Division by zero attempted.")
            raise ValueError("Division by zero is not allowed.")
        print(f"[MathPlugin] Calling divide: {number1} / {number2}")
        return number1 / number2

async def chat_with_plugin_example():
    """Demonstrates a chat interaction where the AI can use a math plugin."""

    kernel = Kernel()

    try:
        api_key = api_key_test
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # --- Create a custom httpx.AsyncClient that does NOT verify SSL ---
        # WARNING: This is INSECURE for production. Use only for testing
        # where you understand the risks or have no other option.
        custom_async_http_client = httpx.AsyncClient(verify=False)

        # Pass this custom httpx.AsyncClient to the OpenAI client
        # The openai.AsyncOpenAI client itself will then be passed to SK's OpenAIChatCompletion
        openai_client_with_custom_http = AsyncOpenAI(
            api_key=api_key,
            http_client=custom_async_http_client # Pass the custom httpx.AsyncClient here
        )

        kernel.add_service(
            OpenAIChatCompletion(
                ai_model_id="gpt-4o-mini",
                # api_key=api_key, # No need to pass api_key directly if using a pre-configured AsyncOpenAI client
                service_id="openai_chat_with_tools",
                async_client=openai_client_with_custom_http, # Pass the pre-configured AsyncOpenAI client here
            ),
        )
        print("OpenAI Chat Completion service added successfully.")
    except Exception as e:
        print(f"Error adding OpenAI service: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable or in a .env file.")
        return

    # --- 2. Add the Plugin to the Kernel ---
    math_plugin = MathPlugin()
    kernel.add_plugin(math_plugin, plugin_name="MathPlugin")
    print("MathPlugin added to the kernel.")

    # Get the chat completion service instance
    chat_service = kernel.get_service(service_id="openai_chat_with_tools")

    # --- 3. Configure Prompt Execution Settings for Function Calling ---
    chat_settings = OpenAIChatPromptExecutionSettings(
        service_id="openai_chat_with_tools",
        temperature=0.7,
        max_tokens=1000,
        tool_choice="auto"
    )

    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful AI assistant. You have access to a MathPlugin to perform calculations. "
        "Use the math functions when the user asks for calculations."
    )

    print("\n--- Chat with Math Plugin Example ---")
    print("Assistant: Hi! I can help you with math. What calculation do you need?")

    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        chat_history.add_user_message(user_input)

        try:
            print("Assistant: ", end="")
            full_response_content = ""

            async for content in chat_service.get_streaming_chat_message_contents(
                chat_history=chat_history,
                settings=chat_settings,
                kernel=kernel
            ):
                if content.content:
                    print(content.content, end="")
                    full_response_content += content.content
            print()

            if full_response_content:
                chat_history.add_assistant_message(full_response_content)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

async def direct_plugin_invocation_example():
    """Demonstrates directly invoking a plugin function without AI orchestration."""

    kernel = Kernel()

    try:
        api_key = api_key_test
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # Apply the same httpx.AsyncClient configuration here if this example
        # also needs to bypass SSL verification.
        custom_async_http_client = httpx.AsyncClient(verify=False)
        openai_client_with_custom_http = AsyncOpenAI(
            api_key=api_key,
            http_client=custom_async_http_client
        )

        kernel.add_service(
            OpenAIChatCompletion(
                ai_model_id="gpt-4o-mini",
                # api_key=api_key, # No need for api_key directly
                service_id="direct_invoke_service",
                async_client=openai_client_with_custom_http,
            ),
        )
    except Exception as e:
        print(f"Error adding OpenAI service for direct invocation: {e}")
        return

    math_plugin = MathPlugin()
    kernel.add_plugin(math_plugin, plugin_name="MyMathPlugin")

    print("\n--- Direct Plugin Invocation Example ---")
    print("Invoking MyMathPlugin.add directly:")

    try:
        add_function = kernel.plugins["MyMathPlugin"]["add"]
        result = await kernel.invoke(add_function, number1=10, number2=5)
        print(f"10 + 5 = {result.value}")

        print("\nInvoking MyMathPlugin.divide with division by zero:")
        try:
            result = await kernel.invoke(add_function, number1=10, number2=0)
            print(f"10 / 0 = {result.value}")
        except ValueError as ve:
            print(f"Caught expected error: {ve}")

    except Exception as e:
        print(f"An error occurred during direct invocation: {e}")


async def main():
    await chat_with_plugin_example()
    await direct_plugin_invocation_example()

if __name__ == "__main__":
    asyncio.run(main())
