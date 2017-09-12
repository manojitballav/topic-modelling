from openpyxl import load_workbook

workbook = load_workbook('Mi_Max_2_amazon.xlsx')
first_sheet = workbook.get_sheet_names()[0]
worksheet = workbook.get_sheet_by_name(first_sheet)

for row in worksheet:
	for cell in row:
		print(cell.value)