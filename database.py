"""
Database module for ClimaSense
Handles historical weather data storage and retrieval
"""
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd
import json


class WeatherDatabase:
    """
    SQLite database for weather data management
    Stores historical data, predictions, and user queries
    """
    
    def __init__(self, db_path: str = 'weather_data.db'):
        """Initialize database connection"""
        self.db_path = db_path
        self.conn = None
        self.create_tables()
    
    def get_connection(self):
        """Get database connection (singleton pattern)"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
        return self.conn
    
    def create_tables(self):
        """Create all necessary database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table 1: Cities - Store city information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                country TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, country)
            )
        ''')
        
        # Table 2: Historical Weather Data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                temperature REAL NOT NULL,
                humidity REAL NOT NULL,
                feels_like REAL,
                pressure REAL,
                wind_speed REAL,
                description TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (city_id) REFERENCES cities (id),
                UNIQUE(city_id, timestamp)
            )
        ''')
        
        # Table 3: Predictions - Store ML predictions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city_id INTEGER NOT NULL,
                prediction_time TIMESTAMP NOT NULL,
                target_time TIMESTAMP NOT NULL,
                predicted_temp REAL NOT NULL,
                predicted_humidity REAL NOT NULL,
                confidence_lower REAL,
                confidence_upper REAL,
                model_used TEXT NOT NULL,
                actual_temp REAL,
                actual_humidity REAL,
                error_temp REAL,
                error_humidity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (city_id) REFERENCES cities (id)
            )
        ''')
        
        # Table 4: Model Performance - Track model accuracy
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                data_points INTEGER NOT NULL,
                temperature_mae REAL,
                temperature_rmse REAL,
                humidity_mae REAL,
                humidity_rmse REAL,
                training_time REAL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (city_id) REFERENCES cities (id)
            )
        ''')
        
        # Table 5: User Queries - Track user searches
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city_name TEXT NOT NULL,
                query_type TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                response_time REAL,
                success BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_historical_city_time 
            ON historical_weather(city_id, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_predictions_city_target 
            ON predictions(city_id, target_time)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_queries_created 
            ON user_queries(created_at)
        ''')
        
        conn.commit()
        print("✓ Database tables created successfully")
    
    # ==================== CITY OPERATIONS ====================
    
    def add_city(self, name: str, country: str, latitude: float, longitude: float) -> int:
        """Add or get city ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Try to get existing city
        cursor.execute('''
            SELECT id FROM cities WHERE name = ? AND country = ?
        ''', (name, country))
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Insert new city
        cursor.execute('''
            INSERT INTO cities (name, country, latitude, longitude)
            VALUES (?, ?, ?, ?)
        ''', (name, country, latitude, longitude))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_city(self, name: str, country: str) -> Optional[Dict]:
        """Get city information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, country, latitude, longitude
            FROM cities WHERE name = ? AND country = ?
        ''', (name, country))
        
        result = cursor.fetchone()
        if result:
            return {
                'id': result[0],
                'name': result[1],
                'country': result[2],
                'latitude': result[3],
                'longitude': result[4]
            }
        return None
    
    # ==================== HISTORICAL DATA OPERATIONS ====================
    
    def add_historical_data(self, city_id: int, data: List[Dict]):
        """
        Add historical weather data
        data format: [{'timestamp': datetime, 'temperature': float, 'humidity': float, ...}, ...]
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for record in data:
            cursor.execute('''
                INSERT OR REPLACE INTO historical_weather 
                (city_id, timestamp, temperature, humidity, feels_like, pressure, wind_speed, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                city_id,
                record['timestamp'],
                record['temperature'],
                record['humidity'],
                record.get('feels_like'),
                record.get('pressure'),
                record.get('wind_speed'),
                record.get('description')
            ))
        
        conn.commit()
        print(f"✓ Added {len(data)} historical records for city {city_id}")
    
    def get_historical_data(self, city_id: int, hours: int = 168) -> pd.DataFrame:
        """
        Get historical weather data for a city
        Returns pandas DataFrame with temperature and humidity
        """
        conn = self.get_connection()
        
        # Get data from last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = '''
            SELECT timestamp, temperature, humidity
            FROM historical_weather
            WHERE city_id = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=(city_id, cutoff_time))
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def has_recent_data(self, city_id: int, hours: int = 24) -> bool:
        """Check if we have recent historical data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute('''
            SELECT COUNT(*) FROM historical_weather
            WHERE city_id = ? AND timestamp >= ?
        ''', (city_id, cutoff_time))
        
        count = cursor.fetchone()[0]
        return count >= 24  # Need at least 24 hours of data
    
    # ==================== PREDICTION OPERATIONS ====================
    
    def save_prediction(self, city_id: int, prediction_data: Dict):
        """
        Save ML prediction to database
        prediction_data: {
            'target_time': datetime,
            'predicted_temp': float,
            'predicted_humidity': float,
            'confidence_lower': float,
            'confidence_upper': float,
            'model_used': str
        }
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (city_id, prediction_time, target_time, predicted_temp, predicted_humidity,
             confidence_lower, confidence_upper, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            city_id,
            datetime.now(),
            prediction_data['target_time'],
            prediction_data['predicted_temp'],
            prediction_data['predicted_humidity'],
            prediction_data.get('confidence_lower'),
            prediction_data.get('confidence_upper'),
            prediction_data['model_used']
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def update_prediction_actual(self, prediction_id: int, actual_temp: float, actual_humidity: float):
        """Update prediction with actual values and calculate error"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get prediction
        cursor.execute('SELECT predicted_temp, predicted_humidity FROM predictions WHERE id = ?', 
                      (prediction_id,))
        result = cursor.fetchone()
        
        if result:
            predicted_temp, predicted_humidity = result
            error_temp = abs(actual_temp - predicted_temp)
            error_humidity = abs(actual_humidity - predicted_humidity)
            
            cursor.execute('''
                UPDATE predictions
                SET actual_temp = ?, actual_humidity = ?, error_temp = ?, error_humidity = ?
                WHERE id = ?
            ''', (actual_temp, actual_humidity, error_temp, error_humidity, prediction_id))
            
            conn.commit()
            print(f"✓ Updated prediction {prediction_id} with actual values")
    
    def get_prediction_accuracy(self, city_id: int, days: int = 7) -> Dict:
        """Calculate prediction accuracy for a city"""
        conn = self.get_connection()
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        query = '''
            SELECT 
                COUNT(*) as total_predictions,
                AVG(error_temp) as avg_temp_error,
                AVG(error_humidity) as avg_humidity_error,
                MIN(error_temp) as min_temp_error,
                MAX(error_temp) as max_temp_error
            FROM predictions
            WHERE city_id = ? AND prediction_time >= ? AND actual_temp IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, conn, params=(city_id, cutoff_time))
        
        if df.empty or df['total_predictions'].iloc[0] == 0:
            return None
        
        return {
            'total_predictions': int(df['total_predictions'].iloc[0]),
            'avg_temp_error': float(df['avg_temp_error'].iloc[0]),
            'avg_humidity_error': float(df['avg_humidity_error'].iloc[0]),
            'min_temp_error': float(df['min_temp_error'].iloc[0]),
            'max_temp_error': float(df['max_temp_error'].iloc[0])
        }
    
    # ==================== MODEL PERFORMANCE OPERATIONS ====================
    
    def save_model_performance(self, city_id: int, model_data: Dict):
        """Save model training performance metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance
            (city_id, model_name, data_points, temperature_mae, temperature_rmse,
             humidity_mae, humidity_rmse, training_time, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            city_id,
            model_data['model_name'],
            model_data['data_points'],
            model_data.get('temperature_mae'),
            model_data.get('temperature_rmse'),
            model_data.get('humidity_mae'),
            model_data.get('humidity_rmse'),
            model_data.get('training_time'),
            json.dumps(model_data.get('metadata', {}))
        ))
        
        conn.commit()
        print(f"✓ Saved performance for {model_data['model_name']}")
    
    def get_model_performance_history(self, city_id: int, model_name: str, limit: int = 10) -> List[Dict]:
        """Get recent model performance history"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT temperature_mae, temperature_rmse, humidity_mae, humidity_rmse,
                   training_time, created_at
            FROM model_performance
            WHERE city_id = ? AND model_name = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (city_id, model_name, limit))
        
        results = cursor.fetchall()
        
        return [{
            'temperature_mae': row[0],
            'temperature_rmse': row[1],
            'humidity_mae': row[2],
            'humidity_rmse': row[3],
            'training_time': row[4],
            'created_at': row[5]
        } for row in results]
    
    # ==================== USER QUERY OPERATIONS ====================
    
    def log_user_query(self, city_name: str, query_type: str, response_time: float = None,
                       success: bool = True, ip_address: str = None, user_agent: str = None):
        """Log user query for analytics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_queries
            (city_name, query_type, response_time, success, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (city_name, query_type, response_time, success, ip_address, user_agent))
        
        conn.commit()
    
    def get_popular_cities(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most searched cities"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT city_name, COUNT(*) as query_count
            FROM user_queries
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY city_name
            ORDER BY query_count DESC
            LIMIT ?
        ''', (limit,))
        
        return cursor.fetchall()
    
    def get_query_stats(self, days: int = 7) -> Dict:
        """Get query statistics"""
        conn = self.get_connection()
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        query = '''
            SELECT 
                COUNT(*) as total_queries,
                COUNT(CASE WHEN success = 1 THEN 1 END) as successful_queries,
                AVG(response_time) as avg_response_time,
                COUNT(DISTINCT city_name) as unique_cities
            FROM user_queries
            WHERE created_at >= ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        
        if df.empty:
            return {}
        
        row = df.iloc[0]
        return {
            'total_queries': int(row['total_queries']),
            'successful_queries': int(row['successful_queries']),
            'avg_response_time': float(row['avg_response_time']) if row['avg_response_time'] else 0,
            'unique_cities': int(row['unique_cities'])
        }
    
    # ==================== UTILITY OPERATIONS ====================
    
    def clear_old_data(self, days: int = 30):
        """Clear historical data older than specified days"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        cursor.execute('DELETE FROM historical_weather WHERE timestamp < ?', (cutoff_time,))
        cursor.execute('DELETE FROM predictions WHERE prediction_time < ?', (cutoff_time,))
        cursor.execute('DELETE FROM user_queries WHERE created_at < ?', (cutoff_time,))
        
        deleted = cursor.rowcount
        conn.commit()
        
        print(f"✓ Deleted {deleted} old records")
        return deleted
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        tables = ['cities', 'historical_weather', 'predictions', 'model_performance', 'user_queries']
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            stats[f'{table}_count'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("✓ Database connection closed")


# Singleton instance
_db_instance = None

def get_database() -> WeatherDatabase:
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = WeatherDatabase()
    return _db_instance
