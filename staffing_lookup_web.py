from flask import Flask, render_template, request
from daily_staffing_lookup import get_daily_staffing, format_staffing_report
import duckdb
from datetime import datetime

app = Flask(__name__)

def get_db_connection():
    """Get connection to the DuckDB database."""
    try:
        db_path = 'nursing_home_staffing.db'
        print(f"Attempting to connect to database at: {db_path}")
        return duckdb.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def get_date_range():
    """Get the minimum and maximum dates available in the database."""
    conn = get_db_connection()
    if not conn:
        return None, None
        
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(WORKDATE), MAX(WORKDATE) FROM staffing")
    min_date, max_date = cursor.fetchone()
    conn.close()
    
    # Convert dates from YYYYMMDD to datetime objects
    min_date = datetime.strptime(str(min_date), '%Y%m%d')
    max_date = datetime.strptime(str(max_date), '%Y%m%d')
    
    return min_date, max_date

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    min_date, max_date = get_date_range()
    
    if request.method == 'POST':
        try:
            provider_id = request.form['provider_id']
            date_str = request.form['date']
            
            # Validate date format and convert to YYYYMMDD
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                db_date = date_obj.strftime('%Y%m%d')
            except ValueError:
                error = "Please enter a valid date in YYYY-MM-DD format"
                return render_template('index.html', error=error, min_date=min_date, max_date=max_date)
            
            # Get staffing data
            data = get_daily_staffing(provider_id, db_date)
            if data:
                result = format_staffing_report(data)
            else:
                error = "No data found for the specified provider ID and date"
                
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('index.html', result=result, error=error, min_date=min_date, max_date=max_date)

if __name__ == '__main__':
    app.run(debug=True) 