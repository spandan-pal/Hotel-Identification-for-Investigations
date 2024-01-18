import os
import pandas as pd

hotels = os.listdir("Data_Raw/train_images")

df = []
for hotel in hotels:
    if str(hotel)[0] != ".":
        path = "Data_Raw/train_images" + "/" + str(hotel)
        images = os.listdir(path)
    
        count = 0
        for image in images:
            if str(image)[0] != ".":
                count += 1
        df.append([hotel, count])

df = pd.DataFrame(df, columns=["Hotel", "n_images"])
df = df.sort_values("n_images", ascending=False)
df.to_csv("Support Files/summary.csv", index=False)