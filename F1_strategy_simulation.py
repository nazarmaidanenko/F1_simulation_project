import fastf1 as ff1
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pymc as pm
import time
import scipy.stats as stats

# Enable FastF1 cache
ff1.Cache.enable_cache('Cache')

# Statistical distribution fitting functions
def evaluate_distribution(data, dist, dist_name):
    """
    Evaluate goodness of fit for different probability distributions
    
    Args:
        data (np.array): Tire life data
        dist (scipy.stats distribution): Distribution to test
        dist_name (str): Name of the distribution
    
    Returns:
        dict: Distribution fitting results
    """
    try:
        # Fit the distribution
        params = dist.fit(data)
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.kstest(data, lambda x: dist.cdf(x, *params))
        
        return {
            "name": dist_name,
            "params": params,
            "ks_stat": ks_stat,
            "p_value": p_value
        }
    except Exception as e:
        st.warning(f"Error fitting {dist_name} distribution: {e}")
        return None

def analyze_tire_distributions(race):
    """
    Analyze tire life distributions for different compounds
    
    Args:
        race (FastF1 Session): Race session data
    
    Returns:
        dict: Best-fitting distributions for each tire compound
    """
    # Distributions to test
    distributions = {
        "Normal": stats.norm,
        "Exponential": stats.expon,
        "Gamma": stats.gamma,
        "Log-Normal": stats.lognorm,
        "Weibull": stats.weibull_min
    }
    
    # Prepare data
    data = race.laps
    tyre = data[['Compound', 'TyreLife']].dropna()
    
    # Ensure compounds are standardized
    tyre['Compound'] = tyre['Compound'].str.strip().str.capitalize()
    
    # Store results
    results = {}
    
    # Analyze distributions for each compound
    for compound in tyre['Compound'].unique():
        compound_data = tyre[tyre['Compound'] == compound]['TyreLife'].values
        
        # Increase flexibility for data requirements
        if len(compound_data) < 5:
            st.warning(f"Insufficient data for {compound} compound. Using default distribution.")
            # Provide a default distribution if data is too limited
            results[compound] = {
                "name": "Normal", 
                "params": stats.norm.fit(compound_data) if len(compound_data) > 0 else (0, 1),
                "ks_stat": 0,
                "p_value": 1
            }
            continue
        
        # Evaluate distributions
        best_fit = {"name": None, "params": None, "ks_stat": float('inf')}
        
        for dist_name, dist in distributions.items():
            try:
                result = evaluate_distribution(compound_data, dist, dist_name)
                
                if result and result['ks_stat'] < best_fit['ks_stat']:
                    best_fit = result
            except Exception as e:
                st.warning(f"Could not fit {dist_name} distribution for {compound}: {e}")
        
        # Ensure a distribution is selected
        if best_fit['name'] is None:
            st.warning(f"No suitable distribution found for {compound}. Using Normal distribution.")
            best_fit = {
                "name": "Normal", 
                "params": stats.norm.fit(compound_data),
                "ks_stat": 0,
                "p_value": 1
            }
        
        results[compound] = best_fit
    
    # If no compounds were processed, provide default
    if not results:
        st.error("No tire compounds could be analyzed. Using default distributions.")
        results = {
            "Soft": {"name": "Normal", "params": (20, 2), "ks_stat": 0, "p_value": 1},
            "Medium": {"name": "Normal", "params": (25, 2), "ks_stat": 0, "p_value": 1},
            "Hard": {"name": "Normal", "params": (30, 2), "ks_stat": 0, "p_value": 1}
        }
    
    return results

def generate_tire_life(distribution_params, base_life=20):
    """
    Generate tire life based on fitted distribution
    
    Args:
        distribution_params (dict): Parameters of the best-fitting distribution
        base_life (int): Base tire life
    
    Returns:
        float: Simulated tire life
    """
    if not distribution_params or distribution_params['name'] is None:
        return base_life
    
    dist_name = distribution_params['name']
    params = distribution_params['params']
    
    # Sample from the appropriate distribution
    if dist_name == "Normal":
        return max(stats.norm.rvs(*params), 5)
    elif dist_name == "Log-Normal":
        return max(stats.lognorm.rvs(*params), 5)
    elif dist_name == "Weibull":
        return max(stats.weibull_min.rvs(*params), 5)
    elif dist_name == "Gamma":
        return max(stats.gamma.rvs(*params), 5)
    elif dist_name == "Exponential":
        return max(stats.expon.rvs(*params), 5)
    else:
        return base_life
def load_real_world_data(session, driver_names):
    """
    Load real-world race data for specific drivers
    
    Args:
        session (FastF1 Session): Loaded session data
        driver_names (list): List of driver names to load data for
    
    Returns:
        dict: Real-world race data for each driver
    """
    real_data = {}
    
    for driver_name in driver_names:
        try:
            driver_laps = session.laps.pick_driver(driver_name)
            
            real_lap_times = driver_laps['LapTime'].dt.total_seconds().values
            real_tires = driver_laps['Compound'].values
            
            real_data[driver_name] = {
                'lap_times': real_lap_times,
                'tires': real_tires
            }
        except Exception as e:
            st.warning(f"Could not load data for driver {driver_name}: {e}")
    
    return real_data




def get_pit_stop_time(track_name):
    """
    Get estimated pit stop time based on track characteristics
    
    Args:
        track_name (str): Name of the Grand Prix
    
    Returns:
        float: Estimated pit stop time
    """
    # Typical pit stop times for different tracks (in seconds)
    pit_stop_times = {
        'Australia': 22.5,
        'Saudi Arabia': 24.0,
        'Singapore': 23.5,
        'Monaco': 20.0,
        'Brazil': 23.0,
        'Default': 22.0
    }
    
    return pit_stop_times.get(track_name, pit_stop_times['Default'])

def create_model_graph(model):
    """Generates and returns the Graphviz model representation."""
    return pm.model_to_graphviz(model)


def lap_time_mcmc(driver_name, base_lap_time, tire_type, total_laps, observed_lap_times):
    try:
        # Check if sufficient data exists
        if len(observed_lap_times) < 5:
            st.warning(f"Insufficient lap time data for {driver_name}. Using default parameters.")
            return None, None
        
        # Define base degradation rates for tires
        base_degradation = {
            'Soft': 0.008, 
            'Medium': 0.004, 
            'Hard': 0.001
        }.get(tire_type, 0.006)  # Default if tire type not found
        
        with pm.Model() as model:
            # Prior distributions
            initial_lap_time = pm.Normal(
                f"initial_lap_time_{driver_name}", 
                mu=base_lap_time, 
                sigma=1.0
            )
            degradation_rate = pm.Normal(
                f"degradation_rate_{driver_name}", 
                mu=base_degradation, 
                sigma=0.005
            )
            
            # Generate lap times with degradation
            laps_array = np.arange(total_laps)
            base_times = initial_lap_time + degradation_rate * laps_array
            
            # Add lap-to-lap variation
            lap_variation = pm.Normal(
                f"lap_variation_{driver_name}", 
                mu=0, 
                sigma=0.2, 
                shape=total_laps
            )
            
            # Combine base times with variation
            final_lap_times = pm.Deterministic(
                f"lap_times_{driver_name}", 
                base_times + lap_variation
            )
            
            # Likelihood
            pm.Normal(
                f"observed_lap_times_{driver_name}", 
                mu=final_lap_times, 
                sigma=0.5, 
                observed=observed_lap_times
            )
            
            # Sampling
            trace = pm.sample(
                draws=1000, 
                tune=500, 
                chains=4, 
                target_accept=0.95, 
                return_inferencedata=True, 
                random_seed=42
            )
            
            return trace
        
    except Exception as e:
        st.error(f"MCMC model creation failed for {driver_name}: {e}")
        return None, None


def get_tire_strategy(driver_name, total_laps, track_name, tire_dist_params):
    compounds = ['Soft', 'Medium', 'Hard']  # Fixed order

    # Track-specific strategy adjustments
    track_factors = {
        'Monaco': {'soft_weight': 0.5, 'medium_weight': 0.3, 'hard_weight': 0.2},
        'Singapore': {'soft_weight': 0.4, 'medium_weight': 0.3, 'hard_weight': 0.3},
        'Default': {'soft_weight': 0.3, 'medium_weight': 0.4, 'hard_weight': 0.3}
    }

    factors = track_factors.get(track_name, track_factors['Default'])

    # Initialize tire limits
    tire_limits = {'Soft': 8, 'Medium': 3, 'Hard': 2}

    # Generate weights and filter based on available compounds
    compound_weights = [
        factors.get(f"{compound.lower()}_weight", 0) for compound in compounds
    ]
    compounds, compound_weights = zip(*[
        (compound, weight) for compound, weight in zip(compounds, compound_weights)
        if compound in tire_dist_params
    ])

    # Normalize weights
    total_weight = sum(compound_weights)
    if total_weight > 0:
        compound_weights = [w / total_weight for w in compound_weights]
    else:
        st.error("Compound weights sum to zero. Cannot proceed.")
        return None

    # Strategy generation
    laps_remaining = total_laps
    strategy = []

    while laps_remaining > 0:
        # Remove unavailable compounds
        available_compounds = [
            compound for compound in compounds if tire_limits[compound] > 0
        ]
        available_weights = [
            compound_weights[i] for i, compound in enumerate(compounds) if compound in available_compounds
        ]

        if not available_compounds:
            st.error(f"No tires available for {driver_name}.")
            break

        # Normalize weights for available compounds
        total_weight = sum(available_weights)
        available_weights = [w / total_weight for w in available_weights]

        # Select a compound based on available weights
        current_compound = np.random.choice(available_compounds, p=available_weights)
        tire_life = generate_tire_life(tire_dist_params[current_compound], base_life=20)
        stint_length = min(int(tire_life), laps_remaining)

        strategy.extend([current_compound] * stint_length)
        laps_remaining -= stint_length

        # Decrement tire limit for the used compound
        tire_limits[current_compound] -= 1

    return pd.Series(strategy, name=driver_name)



class RaceSimulator:
    def __init__(self, drivers, track_telemetry, total_laps, track_name, tire_dist_params):
        self.drivers = drivers
        self.track_telemetry = track_telemetry
        self.total_laps = total_laps
        self.track_name = track_name
        self.results = {driver['name']: [] for driver in drivers}
        self.positions = {driver['name']: [(0, 0)] for driver in drivers}
        self.tire_strategies = {
            driver['name']: get_tire_strategy(driver['name'], total_laps, track_name, tire_dist_params)
            for driver in drivers
        }
        self.pit_stop_time = get_pit_stop_time(track_name)

        self.lap_times = {}
        self.observed_lap_times = {}
        
        for driver in drivers:
            simulated_times = np.random.normal(driver['pace'], 0.5, total_laps)  # Simulate observed data
            trace= lap_time_mcmc(
                driver['name'],
                driver['pace'],
                driver['tire'],
                total_laps,
                simulated_times
            )
            self.lap_times[driver['name']] = trace.posterior[f"lap_times_{driver['name']}"].mean(dim=["chain", "draw"]).values
            self.observed_lap_times[driver['name']] = simulated_times
        
        self.pit_stop_cooldown = {driver['name']: 0 for driver in drivers}
        self.race_log = []

    def get_tire_degradation(self, driver_name, tire_type, lap_number):
        base_rates = {'Soft': 0.03, 'Medium': 0.02, 'Hard': 0.015}
        threshold_laps = {'Soft': 15, 'Medium': 25, 'Hard': 35}
        
        base_rate = base_rates[tire_type]
        threshold = threshold_laps[tire_type]
        
        if lap_number <= threshold:
            degradation = base_rate * lap_number
        else:
            degradation = base_rate * threshold + (0.5 * base_rate * (lap_number - threshold))
        
        return min(degradation, 1)  # Ensure degradation is capped at 1

    def simulate_lap(self, driver, lap_number):
        """Simulate a single lap for a driver."""
        driver_name = driver['name']
        current_tire = self.tire_strategies[driver_name][lap_number - 1]
        base_lap_time = self.lap_times[driver_name][lap_number - 1]
        
        degradation = self.get_tire_degradation(driver_name, current_tire, lap_number)
        lap_time = base_lap_time * (1 + degradation)
        
        tire_change = False
        pit_time = 0
        new_tire = None
        
        # Check for tire puncture
        if degradation >= 0.8:
            if np.random.rand() < 0.02:  # 2% chance of a puncture
                new_tire = np.random.choice(['Soft', 'Medium', 'Hard'])
                st.write(f"⚠️ {driver_name} experiences a tire puncture on lap {lap_number} and switches to {new_tire} tires!")
                tire_change = True
                pit_time = self.pit_stop_time + np.random.uniform(20, 25)  # Pit stop time with penalty
                self.tire_strategies[driver_name][lap_number:] = new_tire  # Update future strategy
                self.pit_stop_cooldown[driver_name] = lap_number
                lap_time += pit_time
        
        # Check for a standard pit stop if not already changed due to a puncture
        if not tire_change and (lap_number - self.pit_stop_cooldown[driver_name] >= 12 and degradation > 0.5):
            new_tire = np.random.choice(['Soft', 'Medium', 'Hard'])
            pit_time = self.pit_stop_time + np.random.uniform(-1, 1)  # Track-specific pit stop time with small variation
            st.write(f"🛠 {driver_name} pits on lap {lap_number}, switching to {new_tire} tires")
            lap_time += pit_time
            tire_change = True
            self.pit_stop_cooldown[driver_name] = lap_number
            self.tire_strategies[driver_name][lap_number:] = new_tire  # Update future strategy
        
        # Log the lap details
        self.race_log.append({
            'Lap Number': lap_number,
            'Driver': driver_name,
            'Lap Time': lap_time,
            'Tire': current_tire,
            'Tire Degradation': degradation,
            'Tire Change This Lap': tire_change,
            'Pit Stop Time': pit_time if tire_change else np.nan,
            'Position By Time': None  # Will be filled later
        })
        
        return lap_time

    def run_race(self, session, driver_info):
        plot_placeholder = st.empty()
        
        for lap in range(1, self.total_laps + 1):
            try:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=self.track_telemetry['X'],
                    y=self.track_telemetry['Y'],
                    mode='lines',
                    name='Track',
                    line=dict(color='white', width=2)
                ))

                for driver in self.drivers:
                    lap_time = self.simulate_lap(driver, lap)
                    self.results[driver['name']].append(lap_time)

                    if len(self.track_telemetry) > 0:
                        position_idx = int((len(self.track_telemetry) - 1) * 
                                        (lap / self.total_laps))
                        new_pos = (
                            self.track_telemetry.iloc[position_idx]['X'],
                            self.track_telemetry.iloc[position_idx]['Y']
                        )
                        self.positions[driver['name']].append(new_pos)

                for driver in self.drivers:
                    positions = self.positions[driver['name']]
                    if len(positions) > 1:
                        x, y = zip(*positions)
                        fig.add_trace(go.Scatter(
                            x=x, y=y,
                            mode='markers+lines',
                            name=driver['name'],
                            marker=dict(size=8)
                        ))

                fig.update_layout(
                    title=f"Race Simulation Progress - Lap {lap}/{self.total_laps}",
                    xaxis_title="X Coordinate",
                    yaxis_title="Y Coordinate",
                    template="plotly_dark",
                    showlegend=True
                )

                plot_placeholder.plotly_chart(fig, use_container_width=True)
                time.sleep(0.5)

            except Exception as e:
                st.error(f"Error during lap {lap}: {str(e)}")
                break

        # Create a DataFrame for the race log
        lap_times_df = pd.DataFrame(self.race_log)

        # Calculate cumulative time for each driver
        lap_times_df['Race Time'] = lap_times_df.groupby('Driver')['Lap Time'].cumsum()
        lap_times_df['Position By Time'] = lap_times_df.groupby('Lap Number')['Race Time'].rank(method='min')

        # Determine the winning driver
        total_times = {driver['name']: sum(self.results[driver['name']]) for driver in self.drivers}
        winner = min(total_times, key=total_times.get)
        st.write(f"🏆 The winner is {winner}!")

        # Display tire strategy for the winner
        winner_strategy = self.tire_strategies[winner]
        tire_changes = [winner_strategy[0]]
        last_tire = winner_strategy[0]

        for tire in winner_strategy[1:]:
            if tire != last_tire:
                tire_changes.append(tire)
                last_tire = tire

        st.write(f"{winner}'s tire strategy was:")
        for i, tire in enumerate(tire_changes):
            st.write(f"Tire change {i}: {tire}")
        st.write('Race Logs')
        st.write(lap_times_df)
        # Compare with real-world data using session.laps
        with st.spinner(f"Comparing {winner}'s performance with real-world data..."):
            # Map winner's full name to their code using driver_info
            driver_mapping = {driver['name']: driver['code'] for driver in driver_info}
            try:
                winner_code = driver_mapping[winner]
            except KeyError:
                st.error(f"Could not find the code for {winner}. Ensure the driver's name is correct.")
                return
            
            # Filter session.laps by winner's code
            real_data = session.laps[session.laps['Driver'] == winner_code]
            
            if real_data.empty:
                st.warning(f"No real-world data available for {winner_code}.")
                return
            
        # Calculate real-world metrics
        real_race_time = real_data['LapTime'].dt.total_seconds().sum()
        real_pace = real_data['LapTime'].dt.total_seconds().mean()
        real_tire_strategy = real_data['Compound'].tolist()

        # Filter to include only unique compounds in their order of appearance
        real_tire_strategy_used = []
        for compound in real_tire_strategy:
            if compound not in real_tire_strategy_used:
                real_tire_strategy_used.append(compound)

        # Count the number of laps in real data
        real_lap_count = real_data['LapNumber'].nunique()

        # Simulated metrics
        sim_race_time = total_times[winner]
        sim_pace = np.mean(self.results[winner])
        sim_lap_count = len(self.results[winner])

        # Comparison
        st.write(f"**{winner_code}'s Real vs Simulated Performance**")
        comparison_df = pd.DataFrame({
            'Metric': ['Overall Race Time (s)', 'Average Pace (s/lap)', 'Tire Strategy', 'Number of Laps'],
            'Simulated': [sim_race_time, sim_pace, ", ".join(tire_changes), sim_lap_count],
            'Real': [real_race_time, real_pace, ", ".join(real_tire_strategy_used), real_lap_count]
        })
        st.dataframe(comparison_df)





def get_track_telemetry(session, driver_code):
    try:
        fastest_lap = session.laps.pick_driver(driver_code).pick_fastest()
        return fastest_lap.get_telemetry().add_distance()
    except Exception as e:
        st.error(f"Failed to load telemetry: {str(e)}")
        return pd.DataFrame({'X': [0], 'Y': [0]})

def main():
    st.title("F1 Track Analysis and Race Simulation")
    
    year = st.sidebar.selectbox("Select Year", [2023, 2024], index=0)
    gp_options = {
        'Australia': 5.303,
        'Saudi Arabia': 6.174,
        'Singapore': 5.063,
        'Monaco': 3.337,
    }
    gp = st.sidebar.selectbox("Select Grand Prix", list(gp_options.keys()))
    
    try:
        # Load session data
        session = ff1.get_session(year, gp, 'R')
        session.load()
        
        # Analyze tire distributions
        tire_dist_params = analyze_tire_distributions(session)
        if not tire_dist_params:
            tire_dist_params = {
                'Soft': {"name": "HalfNormal", "params": (20, 2), "ks_stat": 0, "p_value": 1},
                'Medium': {"name": "HalfNormal", "params": (25, 2), "ks_stat": 0, "p_value": 1},
                'Hard': {"name": "HalfNormal", "params": (30, 2), "ks_stat": 0, "p_value": 1}
            }
        
        # Improved driver selection
        drivers = session.drivers
        
        # Ensure we have at least 2 drivers
        if len(drivers) < 2:
            st.error(f"Insufficient drivers for race simulation. Only {len(drivers)} driver(s) found.")
            return
        
        driver_info = []
        for driver in drivers:
            driver_data = session.get_driver(driver)
            driver_code = driver_data.get('Abbreviation', '')
            full_name = driver_data.get('FullName', 'Unknown Driver')
            
            # Calculate driver pace
            try:
                driver_laps = session.laps.pick_driver(driver_code)
                driver_pace = driver_laps['LapTime'].mean().total_seconds() if not driver_laps.empty else 90  # Default pace
            except Exception as e:
                st.warning(f"Could not calculate pace for {full_name}: {e}")
                driver_pace = 90  # Default pace if calculation fails
            
            driver_info.append({
                'code': driver_code,
                'name': full_name,
                'pace': driver_pace
            })
        
        # Retrieve telemetry
        track_telemetry = get_track_telemetry(session, driver_info[0]['code']) if driver_info else pd.DataFrame({'X': [0], 'Y': [0]})
        
        # Driver selection dropdowns
        selected_drivers = st.sidebar.multiselect(
            "Select Drivers (up to 10)",
            [d['code'] for d in driver_info],
            format_func=lambda x: f"{x} - {next(d['name'] for d in driver_info if d['code'] == x)}",
            default=[d['code'] for d in driver_info[:10]]  # Preselect up to 10 drivers
        )

        if len(selected_drivers) < 2:
            st.warning("Please select at least 2 drivers for the simulation.")
            return

        # Prepare driver data for simulation dynamically
        driver_data = [
            {
                'name': next(d['name'] for d in driver_info if d['code'] == driver_code),
                'pace': next(d['pace'] for d in driver_info if d['code'] == driver_code),
                'tire': 'Soft' if i % 3 == 0 else 'Medium' if i % 3 == 1 else 'Hard'  # Cycle through tire types
            }
            for i, driver_code in enumerate(selected_drivers)
        ]
        
        total_laps = st.sidebar.slider("Number of Laps", 20, 70, 50)
        
        # Start the simulation when the button is clicked
        if st.sidebar.button("Run Race Simulation"):

            with st.spinner("Initializing simulation..."):
                simulator = RaceSimulator(
                    drivers=driver_data,
                    track_telemetry=track_telemetry,
                    total_laps=total_laps,
                    track_name=gp,
                    tire_dist_params=tire_dist_params,
                )
                simulator.run_race(session = session,  driver_info = driver_info)  # No need to pass session


    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try selecting different parameters or refreshing the page.")

if __name__ == "__main__":
    main()
