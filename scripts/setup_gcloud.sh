#!/bin/bash

# Google Cloud Setup Script for VTO Product Recontext System
# This script configures all necessary Google Cloud settings for the system to work properly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Google Cloud Setup for VTO System"
echo "======================================"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}Error: gsutil is not installed${NC}"
    echo "Please install Google Cloud SDK from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please run setup_env.sh first to create the .env file"
    exit 1
fi

# Load values from .env file
if [ -f .env ]; then
    # Extract values from .env, ignoring comments and empty lines
    PROJECT_ID=$(grep "^PROJECT_ID=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    BUCKET_NAME=$(grep "^GOOGLE_CLOUD_STORAGE=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    LOCATION=$(grep "^LOCATION=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
    
    # Set default location if not specified
    if [ -z "$LOCATION" ]; then
        LOCATION="us-central1"
    fi
fi

# Check for placeholder values
if [ "$PROJECT_ID" = "your-project-id" ] || [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}Warning: PROJECT_ID not configured in .env file${NC}"
    echo "Please enter your Google Cloud Project ID:"
    read -r PROJECT_ID
    
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: Project ID is required${NC}"
        exit 1
    fi
    
    # Update .env file
    sed -i.bak "s/PROJECT_ID=.*/PROJECT_ID=$PROJECT_ID/" .env
    echo -e "${GREEN}Updated PROJECT_ID in .env file${NC}"
fi

if [ "$BUCKET_NAME" = "your-bucket-name" ] || [ -z "$BUCKET_NAME" ]; then
    echo -e "${YELLOW}Warning: GOOGLE_CLOUD_STORAGE not configured in .env file${NC}"
    echo "Please enter your desired GCS bucket name:"
    read -r BUCKET_NAME
    
    if [ -z "$BUCKET_NAME" ]; then
        echo -e "${RED}Error: Bucket name is required${NC}"
        exit 1
    fi
    
    # Update .env file
    sed -i.bak "s/GOOGLE_CLOUD_STORAGE=.*/GOOGLE_CLOUD_STORAGE=$BUCKET_NAME/" .env
    echo -e "${GREEN}Updated GOOGLE_CLOUD_STORAGE in .env file${NC}"
fi

echo ""
echo "Configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Bucket Name: $BUCKET_NAME"
echo "  Location: $LOCATION"
echo ""

# Step 1: Check if project exists and set it
echo "Step 1: Checking and setting Google Cloud project..."

# First check if project exists
if gcloud projects describe "$PROJECT_ID" &>/dev/null; then
    if gcloud config set project "$PROJECT_ID" 2>/dev/null; then
        echo -e "${GREEN}✓ Project set to: $PROJECT_ID${NC}"
    else
        echo -e "${RED}✗ Failed to set project configuration${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Project '$PROJECT_ID' not found or no access${NC}"
    echo ""
    echo "Available projects in your account:"
    gcloud projects list --format="table(projectId,name,projectNumber)" 2>/dev/null || echo "  (Unable to list projects - check authentication)"
    echo ""
    echo "Solutions:"
    echo "1. Use an existing project from the list above:"
    echo "   - Edit .env file and change PROJECT_ID to an existing project"
    echo "2. Create a new project:"
    echo "   - Run: gcloud projects create your-unique-project-id-123"
    echo "   - Then update PROJECT_ID in .env file"
    echo "3. Check if you're authenticated to the correct Google account:"
    echo "   - Run: gcloud auth list"
    exit 1
fi

# Step 2: Authenticate if needed
echo ""
echo "Step 2: Checking authentication..."
if ! gcloud auth application-default print-access-token &>/dev/null; then
    echo -e "${YELLOW}Authentication required. Opening browser for authentication...${NC}"
    gcloud auth application-default login
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Authentication successful${NC}"
    else
        echo -e "${RED}✗ Authentication failed${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Already authenticated${NC}"
fi

# Step 3: Enable required APIs
echo ""
echo "Step 3: Enabling required Google Cloud APIs..."
echo "This may take a few minutes..."

APIS=(
    "aiplatform.googleapis.com"
    "storage.googleapis.com"
    "generativelanguage.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo -n "  Enabling $api..."
    if gcloud services enable "$api" --project="$PROJECT_ID" 2>/dev/null; then
        echo -e " ${GREEN}✓${NC}"
    else
        # Check if already enabled
        if gcloud services list --enabled --project="$PROJECT_ID" 2>/dev/null | grep -q "$api"; then
            echo -e " ${GREEN}✓ (already enabled)${NC}"
        else
            echo -e " ${RED}✗ Failed${NC}"
            echo -e "${YELLOW}  Try manually: gcloud services enable $api${NC}"
        fi
    fi
done

# Step 4: Create or configure the bucket
echo ""
echo "Step 4: Setting up Google Cloud Storage bucket..."

# Check if bucket exists
if gcloud storage ls "gs://$BUCKET_NAME" &>/dev/null; then
    echo -e "${GREEN}✓ Bucket already exists: gs://$BUCKET_NAME${NC}"
else
    echo "Creating bucket: gs://$BUCKET_NAME in $LOCATION..."
    if gcloud storage buckets create "gs://$BUCKET_NAME" --location="$LOCATION" 2>/dev/null; then
        echo -e "${GREEN}✓ Bucket created successfully${NC}"
    else
        echo -e "${RED}✗ Failed to create bucket${NC}"
        echo "  Bucket name might be taken or you may not have permissions"
        echo "  Try a different bucket name in your .env file"
        echo "  Or check if you have Storage Admin permissions"
        exit 1
    fi
fi

# Step 5: CRITICAL - Disable uniform bucket-level access
echo ""
echo "Step 5: Configuring bucket for public access..."
echo -e "${YELLOW}IMPORTANT: Disabling uniform bucket-level access to allow public URLs${NC}"

if gcloud storage buckets update "gs://$BUCKET_NAME" --no-uniform-bucket-level-access 2>/dev/null; then
    echo -e "${GREEN}✓ Uniform bucket-level access DISABLED${NC}"
    echo "  Result images will be publicly accessible via URLs"
else
    echo -e "${RED}✗ Failed to disable uniform bucket-level access${NC}"
    echo "  You may need to do this manually in the GCP Console"
fi

# Verify the setting
echo ""
echo "Verifying bucket configuration..."
UNIFORM_ACCESS=$(gcloud storage buckets describe "gs://$BUCKET_NAME" --format="value(iamConfiguration.uniformBucketLevelAccess.enabled)" 2>/dev/null)

if [ "$UNIFORM_ACCESS" = "False" ]; then
    echo -e "${GREEN}✓ Bucket configured correctly for public access${NC}"
else
    echo -e "${RED}✗ Warning: Uniform bucket-level access is still enabled${NC}"
    echo "  Result images may not be publicly accessible"
    echo "  Please disable it manually in GCP Console:"
    echo "  1. Go to Cloud Storage > $BUCKET_NAME"
    echo "  2. Click 'PERMISSIONS' tab"
    echo "  3. Click 'SWITCH TO FINE-GRAINED'"
fi

# Step 6: Set default bucket permissions for new objects
echo ""
echo "Step 6: Setting default ACL for public read access..."
if gcloud storage buckets update "gs://$BUCKET_NAME" --default-object-acl=public-read 2>/dev/null; then
    echo -e "${GREEN}✓ Default ACL set to public-read${NC}"
else
    echo -e "${YELLOW}⚠ Could not set default ACL (this is optional)${NC}"
fi

# Step 7: Create bucket subdirectories structure
echo ""
echo "Step 7: Creating bucket directory structure..."
DIRS=("virtual-try-on" "product-recontext" "logs")

for dir in "${DIRS[@]}"; do
    # Create a placeholder file to establish the directory
    echo "Creating directory: $dir/"
    echo "placeholder" | gcloud storage cp - "gs://$BUCKET_NAME/$dir/.keep" 2>/dev/null || true
done
echo -e "${GREEN}✓ Bucket directory structure created${NC}"

# Step 8: Verify API endpoints
echo ""
echo "Step 8: Verifying model endpoints..."
MODEL_ENDPOINT=$(grep "^MODEL_ENDPOINT=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
MODEL_ENDPOINT_PRODUCT=$(grep "^MODEL_ENDPOINT_PRODUCT=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")

echo "  VTO Endpoint: $MODEL_ENDPOINT"
echo "  Product Recontext Endpoint: $MODEL_ENDPOINT_PRODUCT"
echo -e "${GREEN}✓ Model endpoints configured${NC}"

# Step 9: Final verification
echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo -e "${GREEN}✓ Google Cloud configuration completed successfully!${NC}"
echo ""
echo "Summary:"
echo "  • Project: $PROJECT_ID"
echo "  • Bucket: gs://$BUCKET_NAME"
echo "  • Location: $LOCATION"
echo "  • APIs: Enabled"
echo "  • Public Access: Configured"
echo ""
echo "Next steps:"
echo "1. Ensure your Gemini API key is set in .env file"
echo "2. Run a test: python -m bulk_image_processor --dry-run"
echo "3. Process images: python -m bulk_image_processor --csv image_folder/test_mixed_25.csv --pipeline"
echo ""
echo -e "${YELLOW}Important Notes:${NC}"
echo "• Result images will be publicly accessible at:"
echo "  https://storage.googleapis.com/$BUCKET_NAME/..."
echo "• Monitor your API quotas in the GCP Console"
echo "• Check costs regularly in the Billing section"