from datetime import datetime

user_input = input("enter your goal with a deadline separated by colon\n")
input_list = user_input.split(":")

goal = input_list[0]
deadline = input_list[1]

# print(datetime.datetime.strptime(deadline, "%d.%m.%Y"))
# print(type(datetime.datetime.strptime(deadline, "%d.%m.%Y")))
# print(input_list)

deadline_date = datetime.strptime(deadline, "%d.%m.%Y")
today_date = datetime.today()
time_till = deadline_date - today_date

hours_till = int(time_till.total_seconds()/60/60)

# calculate days not to deadline
print(f" time to deadline goal: {goal} is {time_till.days} days" )
