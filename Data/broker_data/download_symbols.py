import pandas as pd

df = pd.read_csv("https://public.fyers.in/sym_details/NSE_CM.csv", header=None)
# print(df.head(3))
# print(df.shape)
# print(df.describe())
# print(df[3].value_counts().head(20))
# print(df[2].value_counts())
# print(df[19].value_counts())
# print(df[20].value_counts())
# for code in [5, 0, 2, 9, 6, 10, 8, 7, 4]:
#     print(code)
#     print(df[df[2] == code][9].head(5))  # column 9 = symbol ticker
#     print()

final_df = df[df[2] == 0]
final_df.to_csv(
    "/Users/sharmanjeurkar/Projects/SequenceAlpha/Data/symbols/Symbol_data.csv",
    index=False,
)
