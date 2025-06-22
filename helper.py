def days_to_units (num_of_days, conversion_unit):
    if conversion_unit == "hours":
        return f"{num_of_days} days are {num_of_days * 24} hours"
    elif conversion_unit == "minutes":
        return f"{num_of_days} days are {num_of_days * 24 * 60} minutes"
    else:
        return "unsupported unit"

def validate_and_execute(days_and_unit_dictionary):
    try:
        user_input_number = int(days_and_unit_dictionary["days"])
        if user_input_number > 0:
            calculated_value = days_to_units(user_input_number, days_and_unit_dictionary["days"]) #??
            print(calculated_value)
        elif user_input_number == 0:
            print("you entered 0, not a valid positive number.")
        else:
            print(" you entered a negative number, no conversion for it.")
    except ValueError:
        print(" Invalid input. Will not be processed.")


user_input_message = " Hey, please enter number of days and conversion unit! \n"