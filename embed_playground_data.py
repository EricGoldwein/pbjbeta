import json
import re

def embed_data_in_html():
    """Embed the real PBJ data directly into the HTML file to avoid CORS issues."""
    
    # Load the generated data
    try:
        with open('playground_data.json', 'r') as f:
            data = json.load(f)
        
        # Convert to JavaScript object
        js_data = json.dumps(data, indent=2)
        
        # Read the HTML file
        with open('pbj_playground.html', 'r') as f:
            html_content = f.read()
        
        # Replace the null data with the real data
        html_content = re.sub(
            r'pbjData = null; // Will be set by the data generation process',
            f'pbjData = {js_data};',
            html_content
        )
        
        # Write the updated HTML file
        with open('pbj_playground.html', 'w') as f:
            f.write(html_content)
        
        print("✅ Data embedded successfully in pbj_playground.html")
        print(f"📊 HPRD Mean: {data['hprd']['mean']:.3f}")
        print(f"📊 HPRD Median: {data['hprd']['median']:.3f}")
        print(f"📊 Contract Mean: {data['contract']['mean']:.1f}%")
        print(f"📊 Contract Median: {data['contract']['median']:.1f}%")
        
    except FileNotFoundError:
        print("❌ playground_data.json not found. Run 'python generate_playground_data.py' first.")
    except Exception as e:
        print(f"❌ Error embedding data: {e}")

if __name__ == "__main__":
    embed_data_in_html()
