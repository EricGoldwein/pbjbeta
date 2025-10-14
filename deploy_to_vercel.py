#!/usr/bin/env python3
"""
Deployment script for Vercel
This script prepares the files for Vercel deployment
"""

import os
import shutil
import subprocess
import sys

def create_vercel_structure():
    """Create the proper file structure for Vercel deployment"""
    
    print("Creating Vercel deployment structure...")
    
    # Ensure we have all required files
    required_files = [
        'facility_495241_flask_app.py',
        'templates/dynamic_facility_dashboard.html',
        'facility_495241_complete_data.csv',
        'facility_495241_provider_info_data.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"ERROR: Missing required files: {missing_files}")
        return False
    
    # Create api directory if it doesn't exist
    if not os.path.exists('api'):
        os.makedirs('api')
        print("Created api/ directory")
    
    # Check if vercel.json exists
    if not os.path.exists('vercel.json'):
        print("ERROR: vercel.json not found. Please create it first.")
        return False
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("ERROR: requirements.txt not found. Please create it first.")
        return False
    
    print("✅ All required files present")
    print("✅ Vercel structure ready")
    
    return True

def deploy_to_vercel():
    """Deploy to Vercel using CLI"""
    
    print("\nDeploying to Vercel...")
    
    try:
        # Check if Vercel CLI is installed
        result = subprocess.run(['vercel', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: Vercel CLI not found. Please install it first:")
            print("npm i -g vercel")
            return False
        
        print("Vercel CLI found")
        
        # Deploy
        print("Starting deployment...")
        result = subprocess.run(['vercel', '--prod'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Deployment successful!")
            print(result.stdout)
            return True
        else:
            print("❌ Deployment failed!")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main deployment function"""
    
    print("🚀 Vercel Deployment Script")
    print("=" * 40)
    
    # Step 1: Create structure
    if not create_vercel_structure():
        print("❌ Structure creation failed")
        return
    
    # Step 2: Ask user if they want to deploy
    deploy = input("\nDo you want to deploy to Vercel now? (y/n): ").lower().strip()
    
    if deploy == 'y':
        if deploy_to_vercel():
            print("\n🎉 Deployment completed successfully!")
            print("\nNext steps:")
            print("1. Check your Vercel dashboard for the deployment URL")
            print("2. Test the deployed application")
            print("3. Set up a custom domain if needed")
        else:
            print("\n❌ Deployment failed. Check the errors above.")
    else:
        print("\n📁 Files are ready for manual deployment.")
        print("Run 'vercel' command to deploy manually.")

if __name__ == "__main__":
    main()
