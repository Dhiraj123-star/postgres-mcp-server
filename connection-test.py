
"""
Test database connection with current configuration
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test database connection"""
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'testdb'),
        'user': os.getenv('DB_USER', 'testuser'),
        'password': os.getenv('DB_PASSWORD', 'testpass')
    }
    
    print("Database Configuration:")
    print(f"Host: {db_config['host']}")
    print(f"Port: {db_config['port']}")
    print(f"Database: {db_config['database']}")
    print(f"User: {db_config['user']}")
    print(f"Password: {'*' * len(db_config['password'])}")
    print("-" * 40)
    
    try:
        print("Attempting to connect...")
        conn = psycopg2.connect(**db_config)
        print("✅ Connection successful!")
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"PostgreSQL version: {version[0]}")
        
        # Check if our tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        print(f"Available tables: {[table[0] for table in tables]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check if Docker container is running: docker-compose ps")
        print("2. Check container logs: docker-compose logs postgres")
        print("3. Verify .env file has correct port (5433)")
        print("4. Try restarting the container: docker-compose restart postgres")
        return False

if __name__ == "__main__":
    print("PostgreSQL Connection Test")
    print("=" * 40)
    test_connection()