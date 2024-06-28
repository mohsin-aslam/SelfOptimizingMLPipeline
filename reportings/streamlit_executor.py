import subprocess
import time

def run_streamlit_app(script_name):
    # try:
    #     subprocess.run(["streamlit", "run", script_name], check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Failed to run Streamlit app: {e}")
    try:
        # Start the Streamlit app
        process = subprocess.Popen(["streamlit", "run", script_name])
        
        # Wait for 2 minutes (120 seconds)
        time.sleep(18000)
        
        # Terminate the Streamlit app
        process.terminate()
        
        # Ensure the process has ended
        process.wait()
        
        print(f"Streamlit app {script_name} terminated after 2 minutes.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run Streamlit app: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    script_name = "airflow_data/dags/ml_project/reportings.py"  # Replace with your Streamlit script name
    run_streamlit_app(script_name)
