import duckdb
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import os

def get_db_connection():
    """Get connection to the DuckDB database."""
    try:
        db_path = 'nursing_home_staffing.db'
        print(f"Attempting to connect to database at: {os.path.abspath(db_path)}")
        print(f"Database file exists: {os.path.exists(db_path)}")
        print(f"Database file size: {os.path.getsize(db_path) if os.path.exists(db_path) else 'N/A'} bytes")
        return duckdb.connect(db_path)
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def get_facility_info(provnum: str) -> Optional[Dict[str, Any]]:
    """Get basic facility information."""
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        query = """
            SELECT DISTINCT
                PROVNUM,
                PROVNAME,
                STATE,
                COUNTY_NAME,
                CITY
            FROM staffing 
            WHERE PROVNUM = ?
            ORDER BY WORKDATE DESC
            LIMIT 1
        """
        
        result = conn.execute(query, (provnum,)).fetchone()
        conn.close()
        
        if result:
            return {
                'ccn': result[0],
                'provider_name': result[1],
                'state': result[2],
                'county': result[3],
                'city': result[4]
            }
        return None
    except Exception as e:
        print(f"Error getting facility info: {str(e)}")
        return None

def get_daily_staffing(provnum: str, date: str) -> Optional[Dict[str, Any]]:
    """Get detailed staffing data for a specific facility on a specific date."""
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        query = """
            WITH daily_data AS (
                SELECT 
                    WORKDATE,
                    CAST(MDSCENSUS AS FLOAT) as census,
                    PROVNAME,
                    CITY,
                    STATE,
                    -- Direct Care Hours
                    CAST(COALESCE(HRS_RNDON, '0') AS FLOAT) as rn_don_hours,
                    CAST(COALESCE(HRS_RNADMIN, '0') AS FLOAT) as rn_admin_hours,
                    CAST(COALESCE(HRS_RN, '0') AS FLOAT) as rn_direct_hours,
                    CAST(COALESCE(HRS_LPNADMIN, '0') AS FLOAT) as lpn_admin_hours,
                    CAST(COALESCE(HRS_LPN, '0') AS FLOAT) as lpn_direct_hours,
                    CAST(COALESCE(HRS_CNA, '0') AS FLOAT) as cna_hours,
                    CAST(COALESCE(HRS_NATRN, '0') AS FLOAT) as nurse_aide_training_hours,
                    CAST(COALESCE(HRS_MEDAIDE, '0') AS FLOAT) as med_aide_hours,
                    -- Contract Staff Hours
                    CAST(COALESCE(HRS_RNDON_CTR, '0') AS FLOAT) as rn_don_contract_hours,
                    CAST(COALESCE(HRS_RNADMIN_CTR, '0') AS FLOAT) as rn_admin_contract_hours,
                    CAST(COALESCE(HRS_RN_CTR, '0') AS FLOAT) as rn_direct_contract_hours,
                    CAST(COALESCE(HRS_LPNADMIN_CTR, '0') AS FLOAT) as lpn_admin_contract_hours,
                    CAST(COALESCE(HRS_LPN_CTR, '0') AS FLOAT) as lpn_direct_contract_hours,
                    CAST(COALESCE(HRS_CNA_CTR, '0') AS FLOAT) as cna_contract_hours,
                    CAST(COALESCE(HRS_NATRN_CTR, '0') AS FLOAT) as nurse_aide_training_contract_hours,
                    CAST(COALESCE(HRS_MEDAIDE_CTR, '0') AS FLOAT) as med_aide_contract_hours,
                    -- Get state averages for comparison
                    (
                        SELECT AVG(
                            (CAST(COALESCE(HRS_RNDON, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RNADMIN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPNADMIN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_CNA, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_NATRN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_MEDAIDE, '0') AS FLOAT) +
                             CAST(COALESCE(HRS_RNDON_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RNADMIN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPNADMIN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_CNA_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_NATRN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_MEDAIDE_CTR, '0') AS FLOAT)
                            ) / NULLIF(CAST(MDSCENSUS AS FLOAT), 0)
                        )
                        FROM staffing 
                        WHERE STATE = STATE AND WORKDATE = WORKDATE 
                        AND CAST(MDSCENSUS AS FLOAT) > 0
                    ) as state_avg_hprd,
                    -- Get facility's 30-day average for comparison
                    (
                        SELECT AVG(
                            (CAST(COALESCE(HRS_RNDON, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RNADMIN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPNADMIN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_CNA, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_NATRN, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_MEDAIDE, '0') AS FLOAT) +
                             CAST(COALESCE(HRS_RNDON_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RNADMIN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_RN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPNADMIN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_LPN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_CNA_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_NATRN_CTR, '0') AS FLOAT) + 
                             CAST(COALESCE(HRS_MEDAIDE_CTR, '0') AS FLOAT)
                            ) / NULLIF(CAST(MDSCENSUS AS FLOAT), 0)
                        )
                        FROM staffing 
                        WHERE PROVNUM = PROVNUM 
                        AND WORKDATE BETWEEN CAST(WORKDATE AS INTEGER) - 30 AND CAST(WORKDATE AS INTEGER)
                        AND CAST(MDSCENSUS AS FLOAT) > 0
                    ) as facility_30day_avg_hprd
                FROM staffing
                WHERE PROVNUM = ? AND WORKDATE = ?
            )
            SELECT * FROM daily_data
        """
        
        result = conn.execute(query, (provnum, int(date))).fetchone()
        conn.close()
        
        if not result:
            return None
            
        # Calculate totals and HPRD
        census = float(result[1])
        if census <= 0:
            return None
            
        # Calculate hours by category
        rn_hours = {
            'direct': {
                'DON': float(result[5]),
                'Admin': float(result[6]),
                'Direct Care': float(result[7]),
                'Total': float(result[5]) + float(result[6]) + float(result[7])
            },
            'contract': {
                'DON': float(result[13]),
                'Admin': float(result[14]),
                'Direct Care': float(result[15]),
                'Total': float(result[13]) + float(result[14]) + float(result[15])
            }
        }
        
        lpn_hours = {
            'direct': {
                'Admin': float(result[8]),
                'Direct Care': float(result[9]),
                'Total': float(result[8]) + float(result[9])
            },
            'contract': {
                'Admin': float(result[16]),
                'Direct Care': float(result[17]),
                'Total': float(result[16]) + float(result[17])
            }
        }
        
        aide_hours = {
            'direct': {
                'CNA': float(result[10]),
                'Training': float(result[11]),
                'Med Aide': float(result[12]),
                'Total': float(result[10]) + float(result[11]) + float(result[12])
            },
            'contract': {
                'CNA': float(result[18]),
                'Training': float(result[19]),
                'Med Aide': float(result[20]),
                'Total': float(result[18]) + float(result[19]) + float(result[20])
            }
        }
        
        # Calculate total hours and HPRD
        total_hours = (rn_hours['direct']['Total'] + rn_hours['contract']['Total'] +
                      lpn_hours['direct']['Total'] + lpn_hours['contract']['Total'] +
                      aide_hours['direct']['Total'] + aide_hours['contract']['Total'])
        
        total_contract_hours = (rn_hours['contract']['Total'] + 
                              lpn_hours['contract']['Total'] + 
                              aide_hours['contract']['Total'])
        
        # Format the date for display
        display_date = datetime.strptime(str(result[0]), '%Y%m%d').strftime('%Y-%m-%d')
        
        return {
            'date': display_date,
            'census': census,
            'facility_name': result[2],
            'city': result[3],
            'state': result[4],
            'rn_hours': rn_hours,
            'lpn_hours': lpn_hours,
            'aide_hours': aide_hours,
            'total_hours': total_hours,
            'total_contract_hours': total_contract_hours,
            'contract_percentage': (total_contract_hours / total_hours * 100) if total_hours > 0 else 0,
            'hprd': total_hours / census if census > 0 else 0,
            'state_avg_hprd': float(result[21] or 0),
            'facility_30day_avg_hprd': float(result[22] or 0)
        }
    except Exception as e:
        print(f"Error getting daily staffing: {str(e)}")
        return None

def format_staffing_report(data: Dict[str, Any]) -> str:
    """Format the staffing data into a simple, direct statement."""
    if not data:
        return "No data available for the specified date."
    
    # Calculate total RN hours (direct + contract)
    total_rn_hours = (data['rn_hours']['direct']['Total'] + 
                     data['rn_hours']['contract']['Total'])
    
    # Format the date nicely
    date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
    formatted_date = date_obj.strftime('%B %d, %Y')
    
    return f"{data['facility_name']} had {total_rn_hours:.1f} RN staffing hours and {data['census']} residents on {formatted_date}."

def main():
    """Main function to run the staffing lookup tool."""
    if len(sys.argv) != 3:
        print("Usage: python daily_staffing_lookup.py <provider_id> <date>")
        print("Example: python daily_staffing_lookup.py 075001 2024-01-15")
        sys.exit(1)
        
    provnum = sys.argv[1]
    date = sys.argv[2]
    
    # Validate date format and convert to YYYYMMDD
    try:
        parsed_date = datetime.strptime(date, '%Y-%m-%d')
        db_date = parsed_date.strftime('%Y%m%d')
    except ValueError:
        print("Error: Date must be in YYYY-MM-DD format")
        sys.exit(1)
    
    # Get facility info
    facility_info = get_facility_info(provnum)
    if not facility_info:
        print(f"Error: Facility with provider ID {provnum} not found")
        sys.exit(1)
        
    # Get daily staffing data
    staffing_data = get_daily_staffing(provnum, db_date)
    if not staffing_data:
        print(f"Error: No staffing data available for {facility_info['provider_name']} on {date}")
        sys.exit(1)
        
    # Print formatted report
    print(format_staffing_report(staffing_data))

if __name__ == "__main__":
    main() 