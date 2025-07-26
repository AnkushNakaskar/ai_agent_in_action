### Please find the execution of file
[pizza_semantic_kernel_execution.py](./pizza_semantic_kernel_execution.py)
* This is ideal example of pizza order and sysmatic kernal function execution
```commandline
Enter your message >>> Hi
Response from agent >>>  Hello! How can I assist you today with your pizza needs?
Enter your message >>> Show me the Menu card
Invoked get_available_pizza function !!
Response from agent >>>  Here's the menu card with the available pizzas:

1. **Bryon's Bigdamaka pizza** - ₹180.76
2. **Gramin's Small Pizza** - ₹129.87
3. **Jaorin's Special Pizza** - ₹239.76

Let me know if you'd like to order one!
Enter your message >>> okay i need to order Bryon Bigdamaka
Response from agent >>>  I'll need to check your pizza wallet balance first to confirm if you have enough funds to order Bryon's Bigdamaka pizza. Could you please provide your wallet password?
Enter your message >>> 12345
Invoked get_pizza_wallet_balance function !!
Response from agent >>>  Your pizza wallet balance is ₹144.34. Unfortunately, that's not enough to order Bryon's Bigdamaka pizza, which costs ₹180.76.

Would you like to order a different pizza instead? Here are your options:

1. **Gramin's Small Pizza** - ₹129.87
2. **Jaorin's Special Pizza** - ₹239.76

Let me know what you'd like to do!
Enter your message >>> okay get me the Gramin
Invoked Pizza order function !!
Response from agent >>>  Your order for **Gramin's Small Pizza** has been placed successfully! Enjoy your meal! If you need anything else, feel free to ask.
Enter your message >>> q
You pressed 'q' exiting the program

Process finished with exit code 0
```
