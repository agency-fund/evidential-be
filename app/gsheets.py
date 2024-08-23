import gspread
import pandas


def read_sheet(url: str, worksheet: str):
    gc = gspread.service_account()
    sheet = gc.open_by_url(url)
    worksheet = sheet.worksheet(worksheet)
    return pandas.DataFrame(worksheet.get_all_records())
