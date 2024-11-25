#%% # Importing the Necceary Libraries  
import os
import pandas as pd
import matplotlib.pyplot as plt
import folium
from openpyxl import load_workbook

#%%# Load the dataset from your file path
file_path = (r'C:\Users\ghoreisb\Box\Oregon State University\0000- Research_OSU\1_Rail_Project'
        r'\3_Flooding\ASOS\Downloaded\Missouri\2010_2024\asos_missouri_2010_2024.csv')

dataframe = pd.read_csv(file_path)

print(dataframe)
#%% # Plotting the Bar Chart
# Clean the 'station' column by removing leading/trailing whitespaces and making all text uppercase
dataframe['station'] = dataframe['station'].str.strip().str.upper()

# Find unique values in the 'station' column and their frequencies
station_counts = dataframe['station'].value_counts()

# Plotting the frequency of stations
plt.figure(figsize=(18, 9))
bars = station_counts.plot(kind='bar')

# Annotate bars with the value counts rotated 90 degrees and reduced font size
for i, count in enumerate(station_counts):
    plt.text(i, count + 5000, str(count), ha='center', rotation=90, fontsize=12)

plt.title('Frequency of Data at Different Stations in the ASOS Dataset (2010-2024) - Missouri', fontsize=14)
plt.xlabel('Station', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()

# Increase the frame from the top by 7.5% (adjustable if needed)
plt.ylim(0, station_counts.max() * 1.15)

# Show the plot
plt.show()

# %% # Generate the Different files for stations 
# Extract unique stations from the dataframe
unique_stations = dataframe['station'].unique()

# Loop through each station, filter the data, and save it as a separate CSV file
for station in unique_stations:
    # Filter the dataframe for the current station
    station_data = dataframe[dataframe['station'] == station]
    
    # Define the output file path
    output_file_path = (fr'C:\Users\ghoreisb\Box\Oregon State University\0000- Research_OSU\1_Rail_Project\3_Flooding'
                        fr'\ASOS\Downloaded\Missouri\2010_2024\stations\{station}_data.csv')
    
    # Save the filtered data to a CSV file
    station_data.to_csv(output_file_path, index=False)

# Print confirmation
print("Files saved successfully.")

# %% # Hourly precipitation 
# Define the directory path where the CSV files are located
directory_path = (r'C:\Users\ghoreisb\Box\Oregon State University\0000- Research_OSU\1_Rail_Project\3_Flooding'
                  r'\ASOS\Downloaded\Missouri\2010_2024\stations')

# Define the output directory for hourly CSV files
output_directory = os.path.join(directory_path, 'hourly')

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
       
        # Load the dataset
        df = pd.read_csv(file_path, low_memory=False)

        # Ensure the 'valid' (timestamp) and 'p01m' (precipitation) columns are present
        if 'valid' in df.columns and 'p01m' in df.columns:
            # Convert 'valid' to datetime
            df['valid'] = pd.to_datetime(df['valid'])

            # Convert 'p01m' to numeric, setting errors='coerce' to handle non-numeric values (turn them into NaNs)
            df['p01m'] = pd.to_numeric(df['p01m'], errors='coerce').fillna(0)

            # Resample to hourly frequency and sum the precipitation values
            df_hourly = df.resample('H', on='valid').sum().reset_index()

            # Define the output file path
            output_file_path = os.path.join(output_directory, f"{filename.split('.')[0]}_hourly.csv")

            # Save the new hourly data to a CSV file
            df_hourly.to_csv(output_file_path, index=False)
            print(f"Hourly data saved for: {filename}")
        else:
            print(f"The required 'valid' or 'p01m' column is missing in the dataset: {filename}")

print("Processing completed for all files.")


# %% # AEP
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the output directory where the hourly CSV files are located
hourly_directory = (r'C:\Users\ghoreisb\Box\Oregon State University\0000- Research_OSU\1_Rail_Project\3_Flooding'
                    r'\ASOS\Downloaded\Missouri\2010_2024\stations\hourly')

# Loop through each file in the hourly directory
for filename in os.listdir(hourly_directory):
    if filename.endswith('_hourly.csv'):
        file_path = os.path.join(hourly_directory, filename)
        print(f"Processing file for AEP: {filename}")
        
        # Load the hourly dataset
        df = pd.read_csv(file_path)

        # Ensure the 'p01m' column is present and filter out zero values
        if 'p01m' in df.columns:
            df_non_zero = df[df['p01m'] > 0]  # Skip zero values
            
            # Sort the precipitation ('p01m') values in ascending order
            df_sorted = df_non_zero.sort_values(by='p01m', ascending=True).reset_index(drop=True)
            
            # Get the total number of events (rows)
            n = len(df_sorted)
            
            # Rank the precipitation values and calculate AEP
            df_sorted['Rank'] = df_sorted.index + 1
            df_sorted['AEP'] = (n - df_sorted['Rank']) / (n + 1)

            # Plot the scatter AEP curve with constant size and color
            plt.figure(figsize=(10, 6))
            
            # Scatter plot with constant size and color, and higher zorder
            plt.scatter(df_sorted['p01m'], df_sorted['AEP'], s=50, c='blue', alpha=0.7, zorder=5)  # zorder=5 for higher layering

            # Logarithmic scale on the x-axis
            plt.xscale('log')
            plt.title(f'Annual Exceedance Probability (AEP) for {filename.split("_")[0]} Station', fontsize=14)
            plt.xlabel('Hourly Precipitation (mm) [Log Scale]', fontsize=12)
            plt.ylabel('AEP (Probability)', fontsize=12)
            
            # Set y-axis ticks at every 0.1 and add grid lines on both x and y axes
            plt.yticks(np.arange(0, 1.1, 0.1))  # Ticks from 0 to 1 with a step of 0.1
            plt.grid(True, which='both', zorder=1)  # Lower zorder for grid

            plt.tight_layout()

            # Show the plot
            plt.show()
        else:
            print(f"'p01m' column is missing in the dataset: {filename}")

print("AEP processing completed for all files.")


# %% # Generate the Interactive map with Stations on It

# Find unique values in the 'station', 'lon', 'lat' columns and their frequencies
station_counts = dataframe.groupby(['station', 'lon', 'lat']).size().reset_index(name='frequency')

# Create a base map
m = folium.Map(location=[dataframe['lat'].mean(), dataframe['lon'].mean()], zoom_start=6)

# Add each station as a point on the map
for _, row in station_counts.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"Station: {row['station']}\nFrequency: {row['frequency']}",
        tooltip=row['station']
    ).add_to(m)

# Save the map as an HTML file or display it
m.save('station_map.html')
m

# %% # Geberating the Heatmap

# Clean the 'valid' column to datetime format
dataframe['valid'] = pd.to_datetime(dataframe['valid'])

# Create a new column for the hour and the date
dataframe['date'] = dataframe['valid'].dt.date
dataframe['hour'] = dataframe['valid'].dt.floor('H')  # Round down to the hour

# Create a pivot table to count occurrences for each station at each hour
pivot_table = dataframe.pivot_table(index='station', 
                                     columns='hour', 
                                     values='valid', 
                                     aggfunc='count', 
                                     fill_value=0)

# To visualize the pivot table as a heatmap
plt.figure(figsize=(16, 8))
plt.imshow(pivot_table, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(label='Frequency')
plt.title('Frequency of Observations per Station (2010-2024)', fontsize=16)
plt.xlabel('Hour', fontsize=12)
plt.ylabel('Station', fontsize=12)

# Set x-ticks with formatted dates
plt.xticks(ticks=range(len(pivot_table.columns)), 
           labels=[date.strftime('%m/%d/%Y %H:%M') for date in pivot_table.columns], 
           rotation=90)

# Set y-ticks for station names
plt.yticks(ticks=range(len(pivot_table.index)), labels=pivot_table.index)

plt.tight_layout()
plt.show()

# Optional: Save the pivot table to a CSV file for further analysis
pivot_table.to_csv('station_frequency_matrix_2010_2024.csv')


# %%
import pandas as pd

# Load the dataset
file_path = (r'C:\Users\ghoreisb\Box\Oregon State University\0000- Research_OSU\1_Rail_Project'
             r'\3_Flooding\ASOS\Downloaded\Missouri\2010_2024\stations\hourly\D_Min.csv')
data = pd.read_csv(file_path)

# Convert the 'valid' column to datetime format for easier manipulation
data['valid'] = pd.to_datetime(data['valid'])

# Convert 'p01m' to numeric, setting errors='coerce' to handle non-numeric values
data['p01m'] = pd.to_numeric(data['p01m'], errors='coerce')

# Fill NaN values in 'p01m' with 0, assuming they represent no rainfall
data['p01m'].fillna(0, inplace=True)

# Initialize variables to track rainfall events
events = []
current_event_intensity = 0
current_event_start = None
current_event_end = None
current_event_duration = 0
zero_precip_duration = 0
within_event = False  # Flag to indicate if currently within an event

# Define the minimum Inter-Event Time Definition (IETD) in hours
IETD = 4

# Loop through the dataset to correctly identify events
for i in range(len(data)):
    current_time = data.loc[i, 'valid']
    current_rainfall = data.loc[i, 'p01m']
    
    if current_rainfall > 0:
        if not within_event:  # Start a new event if not currently within one
            current_event_start = current_time
            within_event = True
        
        current_event_intensity += current_rainfall
        current_event_end = current_time
        current_event_duration += 1  # Increment the event duration in hours
    else:
        # Check if zero-precipitation is within an event (surrounded by precipitation)
        if within_event and (i < len(data) - 1 and data.loc[i + 1, 'p01m'] > 0):
            zero_precip_duration += 1  # Increment zero-precipitation duration if within event
        elif within_event:
            # If zero-precipitation breaks the event due to exceeding IETD or end of data, finalize the event
            if (i == len(data) - 1 or data.loc[i + 1, 'p01m'] == 0) and (current_time - current_event_end).total_seconds() / 3600 > IETD:
                events.append([
                    f"{current_event_start} to {current_event_end}",
                    current_event_intensity,
                    current_event_duration,
                    zero_precip_duration
                ])
                # Reset variables for the next event
                current_event_intensity = 0
                current_event_start = None
                current_event_end = None
                current_event_duration = 0
                zero_precip_duration = 0
                within_event = False

# Convert the events list to a DataFrame
events_df = pd.DataFrame(events, columns=['Event Period', 'Total Precipitation', 'Event Duration (Hours)', 'Zero-Precipitation Time (Hours)'])

# Sort by 'Total Precipitation' in descending order
events_df = events_df.sort_values(by='Total Precipitation', ascending=False).reset_index(drop=True)

# Calculate total number of rows (events)
total_count = len(events_df)

# Add a 'sorted' column based on the sorted rank
events_df['sorted'] = events_df['Total Precipitation'].rank(method='first', ascending=False)

# Calculate AEP (exceedance probability) as rank/total_count
events_df['AEP'] = events_df['sorted'] / total_count

# Define the path for saving the final DataFrame with all columns
final_output_file_path = (r'C:\Users\ghoreisb\Box\Oregon State University\0000- Research_OSU\1_Rail_Project'
                          r'\3_Flooding\ASOS\Downloaded\Missouri\2010_2024\stations\events\Final_Sorted_Rainfall_Events_Summary_D_Min.csv')

# Save the updated DataFrame to a CSV file
events_df.to_csv(final_output_file_path, index=False)

print("Final sorted rainfall events with AEP have been saved to:", final_output_file_path)

# %%
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np

# Initial fit to get baseline parameters
initial_shape, initial_loc, initial_scale = lognorm.fit(events_df['Total Precipitation'], floc=0)

# Adjustments to fine-tune the parameters
shape_adjustment = 0  # Small adjustment to shape (e.g., increase by 0.05)
loc_adjustment = 0  # Usually leave as 0 for log-normal
scale_adjustment = 0  # Small adjustment to scale (e.g., increase by 5)

# Apply adjustments
adjusted_shape = initial_shape + shape_adjustment
adjusted_loc = initial_loc + loc_adjustment
adjusted_scale = initial_scale + scale_adjustment

# Generate model values using the adjusted parameters
x_fit_lognorm = np.linspace(min(events_df['Total Precipitation']), max(events_df['Total Precipitation']), 100)
y_fit_lognorm_exceedance = 1 - lognorm.cdf(x_fit_lognorm, adjusted_shape, loc=adjusted_loc, scale=adjusted_scale)

# Plot the data and the adjusted log-normal fit
plt.figure(figsize=(10, 6))
plt.scatter(events_df['Total Precipitation'], events_df['AEP'], color='blue', label='Precipitation')
plt.plot(x_fit_lognorm, y_fit_lognorm_exceedance, color='green', label='Adjusted Log-Normal Model')
plt.xlabel('Total Precipitation')
plt.ylabel('Annual Exceedance Curve')
plt.title('Log-Normal Model for Annual Exceedance Probability Curve')
plt.legend()
plt.grid(True)

# Display the adjusted parameters
# equation_text_lognorm = (f"Adjusted Log-Normal params:\n"
#                          f"shape={adjusted_shape:.2f}, loc={adjusted_loc:.2f}, scale={adjusted_scale:.2f}")
# plt.text(0.05, 0.95, equation_text_lognorm, transform=plt.gca().transAxes, fontsize=12,
#          verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.show()
# %%
import matplotlib.pyplot as plt

# Define time intervals in minutes
arrival_start = 0  # 7:45 A.M.
service_start = 15  # 8:00 A.M.
queue_dissipate = 45  # 8:30 A.M.

# Arrival and service rates
arrival_rate = 4  # vehicles per minute
service_rate = 6  # vehicles per minute

# Define time range from 7:45 A.M. to 8:30 A.M.
time = list(range(0, queue_dissipate + 1))  # Total 45 minutes

# Calculate cumulative arrivals and departures over time
arrivals = [arrival_rate * min(t, service_start) + arrival_rate * max(0, t - service_start) for t in time]
departures = [0] * service_start + [service_rate * (t - service_start) for t in time if t >= service_start]

# Calculate queue length over time
queue_length = [arrivals[t] - departures[t] if t >= service_start else arrivals[t] for t in range(len(time))]

# Plotting the diagram
plt.figure(figsize=(10, 6))

# Plot cumulative arrivals
plt.plot(time, arrivals, label="Cumulative Arrivals", color="blue", linestyle='--')

# Plot cumulative departures
plt.plot(time, departures, label="Cumulative Departures", color="green", linestyle='-.')

# Plot queue length
plt.plot(time, queue_length, label="Queue Length", color="red")

# Adding labels and title
plt.xlabel("Time (minutes from 7:45 A.M.)")
plt.ylabel("Number of Vehicles")
plt.title("Toll Booth Queue Dynamics from 7:45 A.M. to 8:30 A.M.")
plt.axvline(x=service_start, color="gray", linestyle=":", label="Service Start (8:00 A.M.)")
plt.axvline(x=queue_dissipate, color="black", linestyle=":", label="Queue Dissipates (8:30 A.M.)")

# Adding legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()

# %%
