"""
Database Manager for Advanced Image Biophysics
Handles PostgreSQL database operations for storing analysis results and metadata
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import warnings

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    import psycopg2.pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    warnings.warn("psycopg2 not available - database functionality disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class DatabaseManager:
    """
    Manages PostgreSQL database connections and operations for microscopy analysis data
    """
    
    def __init__(self):
        self.connection_pool = None
        self.database_available = PSYCOPG2_AVAILABLE and self._check_database_connection()
        
        if self.database_available:
            self._initialize_connection_pool()
            self._create_tables()
    
    def _check_database_connection(self) -> bool:
        """Check if database connection is available"""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return False
            
            # Test connection
            conn = psycopg2.connect(database_url)
            conn.close()
            return True
            
        except Exception:
            return False
    
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            database_url = os.getenv('DATABASE_URL')
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,  # min and max connections
                database_url
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize database connection pool: {str(e)}")
            self.database_available = False
    
    def _get_connection(self):
        """Get a connection from the pool"""
        if not self.database_available or not self.connection_pool:
            raise RuntimeError("Database not available")
        
        return self.connection_pool.getconn()
    
    def _return_connection(self, conn):
        """Return a connection to the pool"""
        if self.connection_pool:
            self.connection_pool.putconn(conn)
    
    def _create_tables(self):
        """Create database tables for microscopy analysis data"""
        
        if not self.database_available:
            return
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id SERIAL PRIMARY KEY,
                    experiment_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    user_id VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'active'
                )
            """)
            
            # Files table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id SERIAL PRIMARY KEY,
                    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                    filename VARCHAR(500) NOT NULL,
                    file_hash VARCHAR(64) UNIQUE,
                    file_size BIGINT,
                    file_format VARCHAR(50),
                    upload_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    dimensions JSONB,
                    pixel_size REAL,
                    time_interval REAL,
                    channels INTEGER,
                    frames INTEGER,
                    file_path TEXT
                )
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id SERIAL PRIMARY KEY,
                    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                    analysis_type VARCHAR(100) NOT NULL,
                    analysis_method VARCHAR(100),
                    parameters JSONB,
                    results JSONB,
                    execution_time REAL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'completed',
                    error_message TEXT
                )
            """)
            
            # Nuclear analysis results table (specialized)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nuclear_analysis (
                    id SERIAL PRIMARY KEY,
                    analysis_result_id INTEGER REFERENCES analysis_results(id) ON DELETE CASCADE,
                    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                    analysis_subtype VARCHAR(100), -- binding, chromatin, elasticity
                    free_fraction REAL,
                    bound_fraction REAL,
                    d_free REAL,
                    d_bound REAL,
                    binding_ratio REAL,
                    mobility_ratio REAL,
                    chromatin_state VARCHAR(50),
                    condensation_index REAL,
                    youngs_modulus REAL,
                    strain_magnitude REAL,
                    nuclear_area REAL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) UNIQUE NOT NULL,
                    user_id VARCHAR(100),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    session_data JSONB,
                    ip_address INET
                )
            """)
            
            # Analysis queue table (for batch processing)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_queue (
                    id SERIAL PRIMARY KEY,
                    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
                    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
                    analysis_type VARCHAR(100) NOT NULL,
                    parameters JSONB,
                    priority INTEGER DEFAULT 5,
                    status VARCHAR(50) DEFAULT 'queued',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    error_message TEXT
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_experiment_id ON files(experiment_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_hash ON files(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_results_file_id ON analysis_results(file_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_results_type ON analysis_results(analysis_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nuclear_analysis_file_id ON nuclear_analysis(file_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON user_sessions(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON analysis_queue(status)")
            
            conn.commit()
            
        except Exception as e:
            if conn:
                conn.rollback()
            warnings.warn(f"Failed to create database tables: {str(e)}")
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def create_experiment(self, name: str, description: str = "", 
                         metadata: Dict[str, Any] = None, user_id: str = None) -> int:
        """Create a new experiment"""
        
        if not self.database_available:
            raise RuntimeError("Database not available")
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiments (experiment_name, description, metadata, user_id)
                VALUES (%s, %s, %s, %s) RETURNING id
            """, (name, description, Json(metadata or {}), user_id))
            
            experiment_id = cursor.fetchone()[0]
            conn.commit()
            return experiment_id
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise RuntimeError(f"Failed to create experiment: {str(e)}")
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def store_file_metadata(self, experiment_id: int, file_info: Dict[str, Any]) -> int:
        """Store file metadata in database"""
        
        if not self.database_available:
            raise RuntimeError("Database not available")
        
        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(file_info)
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if file already exists
            cursor.execute("SELECT id FROM files WHERE file_hash = %s", (file_hash,))
            existing = cursor.fetchone()
            
            if existing:
                return existing[0]
            
            cursor.execute("""
                INSERT INTO files (
                    experiment_id, filename, file_hash, file_size, file_format,
                    metadata, dimensions, pixel_size, time_interval, channels, frames
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """, (
                experiment_id,
                file_info.get('filename', ''),
                file_hash,
                file_info.get('file_size', 0),
                file_info.get('format_type', ''),
                Json(file_info.get('metadata', {})),
                Json(file_info.get('dimensions', {})),
                file_info.get('pixel_size', 0.0),
                file_info.get('time_interval', 0.0),
                file_info.get('channels', 1),
                file_info.get('frames', 1)
            ))
            
            file_id = cursor.fetchone()[0]
            conn.commit()
            return file_id
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise RuntimeError(f"Failed to store file metadata: {str(e)}")
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def store_analysis_results(self, file_id: int, experiment_id: int,
                             analysis_type: str, results: Dict[str, Any],
                             parameters: Dict[str, Any] = None,
                             execution_time: float = 0.0) -> int:
        """Store analysis results in database"""
        
        if not self.database_available:
            raise RuntimeError("Database not available")
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analysis_results (
                    file_id, experiment_id, analysis_type, analysis_method,
                    parameters, results, execution_time, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """, (
                file_id,
                experiment_id,
                analysis_type,
                results.get('method', analysis_type),
                Json(parameters or {}),
                Json(results),
                execution_time,
                results.get('status', 'completed')
            ))
            
            analysis_id = cursor.fetchone()[0]
            
            # Store specialized nuclear analysis data if applicable
            if 'nuclear' in analysis_type.lower() and results.get('status') == 'success':
                self._store_nuclear_analysis_data(cursor, analysis_id, file_id, results)
            
            conn.commit()
            return analysis_id
            
        except Exception as e:
            if conn:
                conn.rollback()
            raise RuntimeError(f"Failed to store analysis results: {str(e)}")
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def _store_nuclear_analysis_data(self, cursor, analysis_id: int, file_id: int, results: Dict[str, Any]):
        """Store specialized nuclear analysis data"""
        
        # Extract nuclear-specific metrics from results
        binding_kinetics = results.get('binding_kinetics', {})
        chromatin_state = results.get('chromatin_state', {})
        elasticity_metrics = results.get('elasticity_metrics', {})
        
        # Determine analysis subtype
        if 'binding' in results.get('method', '').lower():
            subtype = 'binding'
        elif 'chromatin' in results.get('method', '').lower():
            subtype = 'chromatin'
        elif 'elasticity' in results.get('method', '').lower():
            subtype = 'elasticity'
        else:
            subtype = 'general'
        
        cursor.execute("""
            INSERT INTO nuclear_analysis (
                analysis_result_id, file_id, analysis_subtype,
                free_fraction, bound_fraction, d_free, d_bound,
                binding_ratio, mobility_ratio, chromatin_state,
                condensation_index, youngs_modulus, strain_magnitude, nuclear_area
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            analysis_id,
            file_id,
            subtype,
            binding_kinetics.get('free_fraction'),
            binding_kinetics.get('bound_fraction'),
            binding_kinetics.get('D_free'),
            binding_kinetics.get('D_bound'),
            binding_kinetics.get('binding_ratio'),
            binding_kinetics.get('mobility_ratio'),
            chromatin_state.get('chromatin_state'),
            chromatin_state.get('condensation_index'),
            elasticity_metrics.get('youngs_modulus'),
            elasticity_metrics.get('strain_magnitude'),
            elasticity_metrics.get('nuclear_area')
        ))
    
    def get_experiments(self, user_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve experiments from database"""
        
        if not self.database_available:
            return []
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if user_id:
                cursor.execute("""
                    SELECT id, experiment_name, description, created_at, updated_at, metadata, status
                    FROM experiments 
                    WHERE user_id = %s OR user_id IS NULL
                    ORDER BY created_at DESC LIMIT %s
                """, (user_id, limit))
            else:
                cursor.execute("""
                    SELECT id, experiment_name, description, created_at, updated_at, metadata, status
                    FROM experiments 
                    ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            warnings.warn(f"Failed to retrieve experiments: {str(e)}")
            return []
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def get_experiment_files(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Get all files for an experiment"""
        
        if not self.database_available:
            return []
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT id, filename, file_format, file_size, upload_time,
                       dimensions, pixel_size, time_interval, channels, frames, metadata
                FROM files 
                WHERE experiment_id = %s
                ORDER BY upload_time DESC
            """, (experiment_id,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            warnings.warn(f"Failed to retrieve files: {str(e)}")
            return []
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def get_analysis_results(self, file_id: int = None, experiment_id: int = None,
                           analysis_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve analysis results"""
        
        if not self.database_available:
            return []
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT ar.id, ar.analysis_type, ar.analysis_method, ar.parameters,
                       ar.results, ar.execution_time, ar.created_at, ar.status,
                       f.filename, e.experiment_name
                FROM analysis_results ar
                JOIN files f ON ar.file_id = f.id
                JOIN experiments e ON ar.experiment_id = e.id
                WHERE 1=1
            """
            params = []
            
            if file_id:
                query += " AND ar.file_id = %s"
                params.append(file_id)
            
            if experiment_id:
                query += " AND ar.experiment_id = %s"
                params.append(experiment_id)
            
            if analysis_type:
                query += " AND ar.analysis_type = %s"
                params.append(analysis_type)
            
            query += " ORDER BY ar.created_at DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            warnings.warn(f"Failed to retrieve analysis results: {str(e)}")
            return []
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def get_nuclear_analysis_summary(self, experiment_id: int = None) -> Dict[str, Any]:
        """Get summary statistics for nuclear analysis"""
        
        if not self.database_available:
            return {}
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT 
                    analysis_subtype,
                    COUNT(*) as count,
                    AVG(free_fraction) as avg_free_fraction,
                    AVG(bound_fraction) as avg_bound_fraction,
                    AVG(d_free) as avg_d_free,
                    AVG(d_bound) as avg_d_bound,
                    AVG(binding_ratio) as avg_binding_ratio,
                    AVG(youngs_modulus) as avg_youngs_modulus,
                    COUNT(DISTINCT chromatin_state) as chromatin_states_count
                FROM nuclear_analysis na
                JOIN files f ON na.file_id = f.id
            """
            
            if experiment_id:
                query += " WHERE f.experiment_id = %s"
                cursor.execute(query + " GROUP BY analysis_subtype", (experiment_id,))
            else:
                cursor.execute(query + " GROUP BY analysis_subtype")
            
            results = [dict(row) for row in cursor.fetchall()]
            
            return {
                'summary_by_type': results,
                'total_analyses': sum(row['count'] for row in results)
            }
            
        except Exception as e:
            warnings.warn(f"Failed to retrieve nuclear analysis summary: {str(e)}")
            return {}
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def search_experiments(self, search_term: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Search experiments by name, description, or metadata"""
        
        if not self.database_available:
            return []
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT id, experiment_name, description, created_at, metadata, status
                FROM experiments 
                WHERE (experiment_name ILIKE %s OR description ILIKE %s OR metadata::text ILIKE %s)
            """
            params = [f'%{search_term}%', f'%{search_term}%', f'%{search_term}%']
            
            if user_id:
                query += " AND (user_id = %s OR user_id IS NULL)"
                params.append(user_id)
            
            query += " ORDER BY created_at DESC LIMIT 50"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            warnings.warn(f"Failed to search experiments: {str(e)}")
            return []
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def export_experiment_data(self, experiment_id: int) -> Dict[str, Any]:
        """Export complete experiment data for backup or analysis"""
        
        if not self.database_available:
            return {}
        
        try:
            # Get experiment details
            experiments = self.get_experiments()
            experiment = next((exp for exp in experiments if exp['id'] == experiment_id), None)
            
            if not experiment:
                return {}
            
            # Get files and analysis results
            files = self.get_experiment_files(experiment_id)
            analysis_results = self.get_analysis_results(experiment_id=experiment_id)
            nuclear_summary = self.get_nuclear_analysis_summary(experiment_id)
            
            return {
                'experiment': experiment,
                'files': files,
                'analysis_results': analysis_results,
                'nuclear_summary': nuclear_summary,
                'export_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            warnings.warn(f"Failed to export experiment data: {str(e)}")
            return {}
    
    def _calculate_file_hash(self, file_info: Dict[str, Any]) -> str:
        """Calculate hash for file deduplication"""
        
        # Use filename, size, and some metadata for hash calculation
        hash_input = f"{file_info.get('filename', '')}{file_info.get('file_size', 0)}"
        
        # Add format-specific metadata
        if file_info.get('metadata'):
            metadata_str = json.dumps(file_info['metadata'], sort_keys=True)
            hash_input += metadata_str[:100]  # First 100 chars of metadata
        
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database usage statistics"""
        
        if not self.database_available:
            return {'database_available': False}
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            stats = {'database_available': True}
            
            # Count records in each table
            tables = ['experiments', 'files', 'analysis_results', 'nuclear_analysis', 'user_sessions']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
            stats['database_size'] = cursor.fetchone()[0]
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM analysis_results 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            stats['analyses_last_24h'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            warnings.warn(f"Failed to get database stats: {str(e)}")
            return {'database_available': False, 'error': str(e)}
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def cleanup_old_sessions(self, days_old: int = 7):
        """Clean up old user sessions"""
        
        if not self.database_available:
            return
        
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM user_sessions 
                WHERE last_activity < NOW() - INTERVAL '%s days'
            """, (days_old,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old sessions")
            
        except Exception as e:
            if conn:
                conn.rollback()
            warnings.warn(f"Failed to cleanup old sessions: {str(e)}")
            
        finally:
            if conn:
                cursor.close()
                self._return_connection(conn)
    
    def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()


# Global database manager instance
db_manager = DatabaseManager()


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    return db_manager