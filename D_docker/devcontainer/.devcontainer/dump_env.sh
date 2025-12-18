#!/bin/bash
# Dump all environment variables into a .env file

# Output file
output_file="/workspaces/toys/.env"

# Clear the output file if it exists
> "$output_file"

# Loop through all environment variables
for var in $(printenv | awk -F= '{print $1}'); do
    echo "$var=$(printenv $var)" >> "$output_file"
done

echo "Environment variables have been dumped to $output_file"
