# Hotel La Quinta Motor Inns (LQM)

# Reading the data
import pandas as pd
lqm = pd.read_excel("LQM.xlsx")
lqm

#Renaming the columns

lqm.columns = lqm.iloc[1].values

lqm = lqm[2:]
lqm

#Part 1
# "Variables Positively Influencing Profits: Inn Pricing, Presence of College Students in the Area. Variables Negatively Influencing Profits: Ratio of State Population per Inn, Square root of Median Income.
# This trend aligns with our initial predictions. Aspects like the pricing of the Inn directly correlate to higher profits, given that we have the leverage to set higher rates. Moreover, an increased concentration of college students in the area is likely to drive up demand, contributing positively to our profit margins.
# On the contrary, factors like Median Income and the ratio of State Population per Inn negatively impact our profit margins. This can be attributed to the higher probability of competition in these regions, with competitor hotels infringing on our ability to dictate pricing, resulting in reduced demand and adversely affecting our profits."

#Part 2
lqm['Profitability'] = 39.05 - (5.41 * lqm['State Population Per Inn (normalized)']) + \
                       (5.86 * lqm['Price (normalized)']) - \
                       (3.09 * lqm['Square Root of Median Income (normalized)']) + \
                       (1.75 * lqm['College Students in Area (normalized)'])

print(lqm.sort_values("Profitability", ascending = False))

# Hotel in Fresno has the highest profitability but also the highest pofitability with a profitability score of 53.379192.
# Hotel in Los Angeles has the lowest profitability with a profitability score of 23.445409.

#Part 3
num_elements = len(lqm)
decision_var = cvx.Variable(num_elements, boolean=True)
hotel_profit = lqm['Profitability'].values
hotel_price = lqm['Price'].values
fund_limit = 10e6

# Set up the problem
optimization_goal = cvx.Maximize(hotel_profit @ decision_var)
boundaries = [hotel_price @ decision_var <= fund_limit]
optimization_problem = cvx.Problem(optimization_goal, boundaries)

# Solve the problem
optimization_problem.solve()

# Display the optimal solution
optimal_selection = lqm.loc[decision_var.value == 1, ['Hotel', 'Location', 'Price']]
max_profit = optimization_goal.value

print("Best Solution:")
print(optimal_selection.to_string(index=False))
print("\nHighest Possible Profit:", max_profit)

#Part 4

# Identifying unique cities
unique_cities = np.unique(lqm['Location'])

# Create a binary matrix representing hotels in each city
hotel_city_matrix = np.array([lqm['Location'] == city for city in unique_cities], dtype=int)

# Decision variables
decision_var = cvx.Variable(num_elements, boolean=True)

# Objective function
optimization_goal = cvx.Maximize(hotel_profit @ decision_var)

#Constraints
fund_limit = 10e6
hotel_costs = lqm['Price'].values
boundaries = [hotel_costs @ decision_var <= fund_limit]
for i in range(len(unique_cities)):
    boundaries.append(hotel_city_matrix[i] @ decision_var <= 2)

# Problem formulation
optimization_problem = cvx.Problem(optimization_goal, boundaries)

# Solve the problem
optimization_problem.solve()

# Extract the optimal solution
optimal_selection_binary = decision_var.value
max_profit = optimization_problem.value

# Print the optimal solution
optimal_selection = lqm[optimal_selection_binary.astype(bool)]
print(optimal_selection[['Hotel', 'Location', 'Price']])

print("\nHighest Possible Profit:", max_profit)

# In the initial strategy, the focus was purely on augmenting the total profitability, sidestepping the concept of city-wide diversification. Consequently, the recommendation leaned heavily towards acquiring hotels primarily from a single city - South Lake Tahoe, California - where profitability was at its peak.
# In contrast, the revised strategy introduced a new parameter restricting the acquisition of hotels in any given city to a maximum of two. This limitation fosters a diversified approach across cities and deters the tendency to over-invest in a single location.
# As a consequence of this shift in approach, the revised optimal strategy suggests a spread of hotel acquisitions across multiple cities, including Eureka, South Lake Tahoe, Fresno, and Los Angeles. Although this resulted in a marginal dip in total profitability to 205.70, it guarantees a more evenly distributed portfolio by considering hotels from varied cities. Such diversification mitigates risks associated with single-location investments and enhances the robustness of the investment strategy.
# Therefore, this adjusted strategy offers a more comprehensive and balanced approach to hotel selection within the prescribed budget. By taking into account both profitability and city-wide diversification, it aligns with LQM's strategy to sidestep over-concentration of investments in any one city.

#Part 6
# The analysis indicates an optimized strategy for hotel selection within budgetary limits, leveraging key variables such as location, price, and star rating to predict profitability, yielding a portfolio worth $269.92. Expansion of the dataset and incorporation of features like customer reviews, amenities, and local attractions could enhance prediction accuracy. Limiting hotel acquisitions to two per city ensured diversified investments. Improving the model further, we propose considering revenue potential, such as occupancy rates, average daily rates, and demand projections, alongside profitability. Incorporating factors like hotel capacity, market saturation, and competitive analysis could yield more realistic recommendations, promoting a well-balanced, profitable, and geographically diversified portfolio.
