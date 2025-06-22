# # from helper import validate_and_execute, user_input_message
# # import os
# # import logging
# # import django
#
# # logger = logging.getLogger("Main")
# # logger.error("Error happened in the app")
#
#
# # or from helper import * # syntax is simpler, just the method or variables in calling code
# # import helper # you need to call with helper.method/variable name
# # import helper as h # give Help an easier name to call
#
# # print(os.name)
#
# # user_input = ""
# #
# # while user_input != "exit":
# #     user_input = input( user_input_message )
# #     days_and_unit = user_input.split(":")
# #     days_and_unit_dictionary = {"days": days_and_unit[0], "unit": days_and_unit[1]}
# #     validate_and_execute(days_and_unit_dictionary)
#
#
# # python built-in Mocules: math, datetime,
#
# #### project
# import openpyxl
#
# inv_file = openpyxl.load_workbook("inventory.xlsx")
# product_list = inv_file["Sheet1"]
#
# products_per_suppler = {}
# total_value_per_supplier = {}
# products_under_10_inv = {}
#
#
# for product_row in range(2, product_list.max_row + 1):  # skip 1 row is title
#     supplier_name = product_list.cell(product_row, 4).value
#     inventory = product_list.cell(product_row, 2).value
#     price = product_list.cell(product_row, 3).value
#     product_num = product_list.cell(product_row, 1).value
#     inventory_price = product_list.cell(product_row, 5)
#
#     # num of products per supplier
#     if supplier_name in products_per_suppler:
#         current_num_products = products_per_suppler[supplier_name]
#         products_per_suppler[supplier_name] = current_num_products + 1
#     else:
#         products_per_suppler[supplier_name] = 1
#
#     # calculate total value of inventory per supplier
#     if supplier_name in total_value_per_supplier:
#         current_total_value = total_value_per_supplier.get(supplier_name)
#         total_value_per_supplier[supplier_name] = current_total_value + inventory * price
#
#     else:
#         total_value_per_supplier[supplier_name] = inventory * price
#
#     # calculate products with inventory less than 10
#     if inventory < 10:
#         products_under_10_inv[int(product_num)] = int(inventory)
#
#     # add value for total inventory price
#     inventory_price.value = inventory * price
#
# #save the added value in a new file
# inv_file.save("inventory_with_total_value.xlsx")
#
#
# print(f"product with inventory <10:   {products_under_10_inv} ")
# print(f"product per supplier: {products_per_suppler}" )
# print(f"total value per supplier: {total_value_per_supplier}" )


#import user
# from user import User # better syntax to directly call imported class methods
# from post import Post
#
#
# user1 = User("nn@nn.com", "nana Jan", "pwd1", "dev")
#
# user1.get_user_info()
#
# new_post = Post ("on a mission today", user1.name )
# new_post.get_post_info()

# make api call
import requests

response = requests.get("https://gitlab.com/api/v4/users/nanuchi/projects")
# print(response.text)
# print(type(response.text))
# print(response.json())
# print(response.json()[0]) # 1st
my_projects = response.json()

for project in my_projects:
    print(f"Project Name: {project['name']} \nProject Url: {project['web_url']}\n")



