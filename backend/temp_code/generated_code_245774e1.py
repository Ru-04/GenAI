# Define a function to print Hello, World!
def print_hello_world():
    print("Hello, World!")


# Define a function to calculate the sum of two numbers
def calculate_sum(a, b):
    return a + b


# Main program
if __name__ == "__main__":
    # Call the print_hello_world function
    print_hello_world()

    # Get user input for a and b
    num1 = int(input("Enter first number: "))
    num2 = int(input("Enter second number: "))

    # Calculate the sum of a and b
    result = calculate_sum(num1, num2)

    # Print the result
    print(f"The sum is: {result}")
