# xRI Asset API (didapi.io)

The **xRI Asset API** allows users to retrieve assets and insights generated from Built Environment Scanning System (BESS) data capture. The API provides secure, authenticated access to binary asset files using a unique identifier and file extension.

## Base URL

```
https://didapi.io/v1
```

## Authentication

All requests must include an API key in the `x-api-key` header for authentication:

```http
x-api-key: YOUR_API_KEY_HERE
```

For the time being, xRI will generate keys on your behalf and will be sent to you via email. Contact support for any API key related questions.

## Endpoint

### Retrieve an Asset

**GET** `/asset/{assetType}/uprn/{uprn}`

#### Path Parameters:

- `assetType` (enum, required): The type of the asset you want to download, choose one of the available options below.
- `uprn` (string, required): The UPRN of the building.

#### Headers:

- `x-api-key` (string, required): Your API authentication key.

#### Asset Types

- "lidar-pointcloud-merged"
- "lidar-range-pano"
- "lidar-reflectance-pano"
- "lidar-signal-pano"
- "lidar-nearir-pano"
- "lidar-pointcloud-frame"
- "ir-false-color-image"
- "ir-temperature-array"
- "ir-count-image"
- "rgb-image"

#### Example Request

```bash
curl -L -O -J "https://didapi.io/v1/asset/lidar-range-pano/uprn/34015415" \
  -H "x-api-key: YOUR_API_KEY_HERE"
```

This will download the asset using the filename suggested by the server in the `Content-Disposition` header.

#### Important

When calling the API you _must allow redirects_. Once your request is successfully authenticated, our server will issue you with a URL redirect to an S3 bucket. This redirect URL will be valid for 60 minutes and is signed with your API key.

## Responses

### Success Response

- **Status:** `200 OK`
- **Body:** Binary content of the requested asset.

### Error Responses

| Status Code | Error                 | Description                                            |
| ----------- | --------------------- | ------------------------------------------------------ |
| `302`       | Found/Redirect        | Moved to another URL.                                  |
| `401`       | Unauthorized          | Missing or invalid API key.                            |
| `404`       | Asset not found       | The requested asset does not exist.                    |
| `429`       | Rate limit exceeded   | Too many requests. Please wait and retry later.        |
| `500`       | Internal server error | An unexpected error occurred while processing request. |

### Error Example

```json
{
  "error": "Asset not found."
}
```

## Rate Limiting

The API implements rate limiting. If your rate limit is exceeded, you'll receive a `429` error. Please contact support to request an increase in your quota if needed.

## Notes

- Ensure your API key is kept secure and never exposed publicly.
- If you key is exposed, message support and we will delete it and recreate a new one.
- Contact support for assistance or to resolve quota-related issues.
