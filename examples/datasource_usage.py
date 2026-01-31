#!/usr/bin/env python3
"""Example usage of the datasource factory with multiple database providers."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_agentic_chatbot.datasource_init import initialize_datasources, get_default_datasource
from ai_agentic_chatbot.infrastructure.datasource import (
    get_engine,
    get_session,
    get_datasource_factory,
    register_mysql_datasource,
    register_postgresql_datasource,
)
from ai_agentic_chatbot.datasource_types import DataSourceType, DataSourceProvider


def demonstrate_datasource_usage():
    """Demonstrate datasource factory usage."""
    print("🗄️  Datasource Factory Usage Examples")
    print("=" * 45)
    
    try:
        # Initialize datasources from config
        print("1. Initializing datasources from config:")
        factory = initialize_datasources()
        
        # Show available datasources
        datasources = factory.list_datasources()
        print(f"   Available datasources: {datasources}")
        
        # Get default datasource
        default_ds = get_default_datasource()
        print(f"   Default datasource: {default_ds}")
        
        # Test connections (will fail with dummy config, but shows the pattern)
        print("\n2. Testing datasource connections:")
        for ds_name in datasources:
            try:
                info = factory.get_datasource_info(ds_name)
                print(f"   {ds_name}: {info['provider']} ({info['host']}:{info['port']})")
                
                # Test connection (will likely fail with dummy credentials)
                connected = factory.test_connection(ds_name)
                status = "✅ Connected" if connected else "❌ Connection failed"
                print(f"     Status: {status}")
                
            except Exception as e:
                print(f"     Error: {e}")
        
        # Demonstrate programmatic registration
        print("\n3. Programmatic datasource registration:")
        
        # Register a MySQL datasource programmatically
        register_mysql_datasource(
            name="temp_mysql",
            host="localhost",
            database="test_db",
            username="test_user",
            password="test_pass",
            ds_type=DataSourceType.CACHE
        )
        print("   ✅ Registered temp_mysql datasource")
        
        # Register a PostgreSQL datasource programmatically
        register_postgresql_datasource(
            name="temp_postgres",
            host="localhost",
            database="test_db",
            username="test_user", 
            password="test_pass",
            ds_type=DataSourceType.ANALYTICS
        )
        print("   ✅ Registered temp_postgres datasource")
        
        # Show updated datasource list
        updated_datasources = factory.list_datasources()
        print(f"   Updated datasources: {updated_datasources}")
        
        # Demonstrate filtering by type and provider
        print("\n4. Filtering datasources:")
        
        primary_datasources = factory.get_datasources_by_type(DataSourceType.PRIMARY)
        print(f"   Primary datasources: {list(primary_datasources.keys())}")
        
        mysql_datasources = factory.get_datasources_by_provider(DataSourceProvider.MYSQL)
        print(f"   MySQL datasources: {list(mysql_datasources.keys())}")
        
        # Demonstrate engine and session usage
        print("\n5. Engine and session usage examples:")
        print("   # Get SQLAlchemy engine")
        print("   engine = get_engine('primary')")
        print("   with engine.connect() as conn:")
        print("       result = conn.execute('SELECT 1')")
        print()
        print("   # Get SQLAlchemy session")
        print("   session = get_session('primary')")
        print("   try:")
        print("       users = session.execute('SELECT * FROM users').fetchall()")
        print("   finally:")
        print("       session.close()")
        
        print("\n✅ Datasource factory demonstration completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_environment_variables():
    """Show environment variable usage for secure credential management."""
    print("\n🔐 Environment Variable Examples")
    print("=" * 35)
    
    print("Set these environment variables for secure credential management:")
    print()
    
    print("# MySQL (Azure Cloud SQL)")
    print("export MYSQL_HOST='your-azure-mysql-host.mysql.database.azure.com'")
    print("export MYSQL_PORT='3306'")
    print("export MYSQL_DATABASE='your_database_name'")
    print("export MYSQL_USERNAME='your_username'")
    print("export MYSQL_PASSWORD='your_password'")
    print()
    
    print("# PostgreSQL")
    print("export POSTGRES_HOST='your-postgres-host.postgres.database.azure.com'")
    print("export POSTGRES_PORT='5432'")
    print("export POSTGRES_DB='your_database_name'")
    print("export POSTGRES_USER='your_username'")
    print("export POSTGRES_PASSWORD='your_password'")
    print()
    
    print("# AWS RDS")
    print("export AWS_RDS_HOST='your-rds-instance.region.rds.amazonaws.com'")
    print("export AWS_RDS_PORT='3306'")
    print("export AWS_RDS_DATABASE='your_database_name'")
    print("export AWS_RDS_USERNAME='your_username'")
    print("export AWS_RDS_PASSWORD='your_password'")
    print()
    
    print("# Azure SQL")
    print("export AZURE_SQL_HOST='your-azure-sql-server.database.windows.net'")
    print("export AZURE_SQL_DATABASE='your_database_name'")
    print("export AZURE_SQL_USERNAME='your_username'")
    print("export AZURE_SQL_PASSWORD='your_password'")


if __name__ == "__main__":
    demonstrate_datasource_usage()
    demonstrate_environment_variables()
