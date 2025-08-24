import os

def get_token():
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        print(f"Current directory (using os): {current_directory}")
        token_path = str(current_directory)+ "/api_key.txt"
        print(token_path)
        with open(token_path, 'r') as file:
            content = file.read();
        print(content)
        return content.strip();
    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_org_id():
    try:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        print(f"Current directory (using os): {current_directory}")
        token_path = str(current_directory)+ "/org_id.txt"
        print(token_path)
        with open(token_path, 'r') as file:
            content = file.read();
        print(content)
        return content.strip();
    except FileNotFoundError:
        print("The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
