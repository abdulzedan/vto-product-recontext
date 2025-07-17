"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters."""
    
    max_workers: int = Field(default=10, ge=1, le=50)
    max_retries: int = Field(default=5, ge=1, le=10)
    download_timeout: int = Field(default=30, ge=5, le=300)
    processing_timeout: int = Field(default=300, ge=60, le=1800)
    batch_size: int = Field(default=100, ge=1, le=1000)


class GoogleCloudConfig(BaseModel):
    """Google Cloud configuration."""
    
    project_id: str = Field(..., min_length=1)
    location: str = Field(default="us-central1")
    model_endpoint: str = Field(default="virtual-try-on-exp-05-31")
    model_endpoint_product: str = Field(default="imagen-product-recontext-preview-06-30")
    storage_bucket: str = Field(..., min_length=1)


class GeminiConfig(BaseModel):
    """Gemini API configuration."""
    
    api_key: str = Field(..., min_length=1)
    model_name: str = Field(default="gemini-1.5-flash")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, ge=1, le=8192)


class StorageConfig(BaseModel):
    """Storage configuration."""
    
    local_output_dir: Path = Field(default=Path("./output"))
    enable_gcs_upload: bool = Field(default=True)
    gcs_virtual_try_on_path: str = Field(default="virtual-try-on")
    gcs_product_recontext_path: str = Field(default="product-recontext")
    gcs_logs_path: str = Field(default="logs")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    log_file: Path = Field(default=Path("./logs/processing.log"))


class Settings(BaseSettings):
    """Main settings class."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Google Cloud settings
    project_id: str = Field(..., validation_alias="PROJECT_ID")
    location: str = Field(default="us-central1", validation_alias="LOCATION")
    model_endpoint: str = Field(
        default="virtual-try-on-exp-05-31", 
        validation_alias="MODEL_ENDPOINT"
    )
    model_endpoint_product: str = Field(
        default="imagen-product-recontext-preview-06-30",
        validation_alias="MODEL_ENDPOINT_PRODUCT"
    )
    google_cloud_storage: str = Field(
        ..., 
        validation_alias="GOOGLE_CLOUD_STORAGE"
    )
    
    # Gemini settings
    gemini_api_key: str = Field(..., validation_alias="GEMINI_API_KEY")
    
    # Processing settings
    max_workers: int = Field(default=10, validation_alias="MAX_WORKERS")
    max_retries: int = Field(default=5, validation_alias="MAX_RETRIES")
    download_timeout: int = Field(default=30, validation_alias="DOWNLOAD_TIMEOUT")
    processing_timeout: int = Field(default=300, validation_alias="PROCESSING_TIMEOUT")
    
    # Storage settings
    local_output_dir: str = Field(default="./output", validation_alias="LOCAL_OUTPUT_DIR")
    enable_gcs_upload: bool = Field(default=True, validation_alias="ENABLE_GCS_UPLOAD")
    
    # Logging settings
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: str = Field(default="json", validation_alias="LOG_FORMAT")
    
    @property
    def processing(self) -> ProcessingConfig:
        """Get processing configuration."""
        return ProcessingConfig(
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            download_timeout=self.download_timeout,
            processing_timeout=self.processing_timeout,
        )
    
    @property
    def google_cloud(self) -> GoogleCloudConfig:
        """Get Google Cloud configuration."""
        return GoogleCloudConfig(
            project_id=self.project_id,
            location=self.location,
            model_endpoint=self.model_endpoint,
            model_endpoint_product=self.model_endpoint_product,
            storage_bucket=self.google_cloud_storage,
        )
    
    @property
    def gemini(self) -> GeminiConfig:
        """Get Gemini configuration."""
        return GeminiConfig(
            api_key=self.gemini_api_key,
        )
    
    @property
    def storage(self) -> StorageConfig:
        """Get storage configuration."""
        return StorageConfig(
            local_output_dir=Path(self.local_output_dir),
            enable_gcs_upload=self.enable_gcs_upload,
        )
    
    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration."""
        return LoggingConfig(
            level=self.log_level,
            format=self.log_format,
        )


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()