#### Chapter 5: Empowering agent with actions
* First we will figure out the function integration in OpenAI 
* File [agent_fuction_print.py](agent_function_print_only/agent_function_print_only.py)
  * This file explain the various functions are getting integrated in open AI calls
  * LLM is using these functions as plugin for various use cases like movie recommendations
  * You can see the function is getting match for recommendation of movie with mandatory param
* File [agent_function_actual_call.py](agent_function_calling_with_sample_function/agent_function_actual_call.py)
  * This file explain the actual function call
  * File have sample python function, which will be use for invocation of LLM agent
* File [semantic_kernel_execution.py](semantic_kernel/semantic_kernel_execution.py)
  * This file explain the semantic kernel library with more details on invocation
  * Please read more on semantic kernel : [here](https://learn.microsoft.com/en-us/semantic-kernel/overview/)
* Check this link kernel working : [here](https://medium.com/@speaktoharisudhan/build-an-agent-orchestrator-in-python-with-semantic-kernel-bb271d8f32e1)

