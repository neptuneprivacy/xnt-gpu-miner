#!/bin/bash
# Test script for XNT-Core Mining RPC Endpoints
# Usage: ./test-mining-rpc.sh [RPC_URL] [GUESSER_ADDRESS]

RPC_URL="${1:-http://127.0.0.1:9899}"
GUESSER_ADDRESS="${2:-xntnwm1test}"

echo "=========================================="
echo "Testing XNT-Core Mining RPC Endpoints"
echo "=========================================="
echo "RPC URL: $RPC_URL"
echo "Guesser Address: $GUESSER_ADDRESS"
echo ""

# Test 1: Check chain height
echo "1. Testing chain_height..."
HEIGHT_RESPONSE=$(curl -s -X POST "$RPC_URL" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"chain_height","params":[],"id":1}')
echo "Response: $HEIGHT_RESPONSE"
echo ""

# Test 2: Check tip digest
echo "2. Testing chain_tipDigest..."
TIP_RESPONSE=$(curl -s -X POST "$RPC_URL" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"chain_tipDigest","params":[],"id":2}')
echo "Response: $TIP_RESPONSE"
echo ""

# Test 3: Check network
echo "3. Testing node_network..."
NETWORK_RESPONSE=$(curl -s -X POST "$RPC_URL" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"node_network","params":[],"id":3}')
echo "Response: $NETWORK_RESPONSE"
echo ""

# Test 4: Get block template (mining endpoint)
echo "4. Testing mining_getBlockTemplate..."
TEMPLATE_RESPONSE=$(curl -s -X POST "$RPC_URL" \
  -H "Content-Type: application/json" \
  -d "{\"jsonrpc\":\"2.0\",\"method\":\"mining_getBlockTemplate\",\"params\":[\"$GUESSER_ADDRESS\"],\"id\":4}")
echo "Response: $TEMPLATE_RESPONSE"
echo ""

# Check if template was received
if echo "$TEMPLATE_RESPONSE" | grep -q '"template"'; then
  echo "✓ Block template received successfully!"
  
  # Extract key fields from template
  echo ""
  echo "Template Summary:"
  echo "$TEMPLATE_RESPONSE" | python3 -m json.tool 2>/dev/null | grep -E "(digest|prevBlock|threshold|totalGuesserReward)" | head -5 || echo "  (Use jq for better formatting)"
else
  echo "✗ Failed to get block template"
  if echo "$TEMPLATE_RESPONSE" | grep -q '"error"'; then
    echo "Error details:"
    echo "$TEMPLATE_RESPONSE" | python3 -m json.tool 2>/dev/null | grep -A 5 '"error"' || echo "$TEMPLATE_RESPONSE"
  fi
fi

echo ""
echo "=========================================="
echo "Test completed"
echo "=========================================="
