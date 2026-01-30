# Test Stratum Server

A simple test server implementing the schema.rs protocol for testing the xnt-gpu-miner stratum client.

## Usage

### Start the server (test jobs only)

```bash
cd test-stratum-server
cargo run
```

### Start the server with RPC (fetch real jobs)

```bash
cd test-stratum-server
RPC_URL="http://127.0.0.1:9897" \
GUESSER_ADDRESS="xnt1your_wallet" \
cargo run
```

### Environment Variables

- `PORT` - Stratum server port (default: 3333)
- `RPC_URL` - JSON-RPC endpoint URL (required for RPC mode)
- `RPC_USER` - RPC username (optional)
- `RPC_PASSWORD` - RPC password (optional)
- `GUESSER_ADDRESS` - Wallet address for mining (default: `xnt1test1234567890abcdef`)
- `RPC_FETCH_INTERVAL` - Job fetch interval in seconds (default: 30)

### Test with the miner

```bash
cd ..
./xnt-miner \
    --wallet "test_address_1234567890abcdef" \
    --stratum "stratum+tcp://127.0.0.1:3333" \
    --stratum-pass "test_password" \
    --stratum-worker "test_worker" \
    --device 0
```

## Protocol Implementation

The server implements the schema.rs protocol:

- **Login**: Accepts login requests and assigns worker IDs
- **Job Notifications**: Sends job notifications after successful login
- **Submit**: Accepts and validates share submissions
- **Keepalive**: Responds to keepalive messages
- **Pause**: (Not yet implemented)

## Features

- Simple test job generation
- Duplicate share detection
- Basic share validation
- Connection tracking
- Logging of all messages

## Testing

The server logs all incoming requests and outgoing responses, making it easy to debug protocol issues.
