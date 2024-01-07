import streamlit as st

class Car:

    def __init__(self, speed=0):
        self.speed = speed
        self.odometer = 0
        self.time = 0

    def accelerate(self):
        self.speed += 5

    def brake(self):
        self.speed -= 5

    def step(self):
        self.odometer += self.speed
        self.time += 1

    def average_speed(self):
        return self.odometer / self.time

def main():
    st.title("Car Interaction with Streamlit")

    # Create a car instance
    my_car = Car()

    # Display the initial state
    st.write("I'm a car!")

    # Counter for creating unique keys
    button_counter = 0

    # Main interaction loop
    while st.button(f"Continue {button_counter}"):
        action = st.radio("What should I do?", ['Accelerate', 'Brake', 'Show Odometer', 'Show Average Speed'])

        if action == 'Accelerate':
            my_car.accelerate()
            st.write("Accelerating...")

        elif action == 'Brake':
            my_car.brake()
            st.write("Braking...")

        elif action == 'Show Odometer':
            st.write("The car has driven {} kilometers".format(my_car.odometer))

        elif action == 'Show Average Speed':
            st.write("The car's average speed was {} kph".format(my_car.average_speed()))

        my_car.step()

        # Increment the button counter for a unique key
        button_counter += 1

if __name__ == '__main__':
    main()
