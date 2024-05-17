
import pandas as pd
import pandas as pd




# Load your dataframe (replace 'your_data.csv' with your actual data file)
df1 = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER1_2cities.csv')
df2 = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER2_2cities.csv')
df3 = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER3_2cities.csv')



import pandas as pd

# Assuming df1, df2, and df3 are your dataframes
# Concatenate them vertically (along rows)
combined_df = pd.concat([df1, df2, df3])

# If you want to reset the index after concatenating
combined_df.reset_index(drop=True, inplace=True)
combined_df.to_csv(f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/GENERAL_6cities.csv', index=False)



# Load your dataframe (replace 'your_data.csv' with your actual data file)
df1 = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER1_5cities.csv')
df2 = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER2_5cities.csv')
df3 = pd.read_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER3_5cities.csv')



import pandas as pd

# Assuming df1, df2, and df3 are your dataframes
# Concatenate them vertically (along rows)
combined_df = pd.concat([df1, df2, df3])

# If you want to reset the index after concatenating
combined_df.reset_index(drop=True, inplace=True)
combined_df.to_csv(f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/GENERAL_15cities.csv', index=False)
