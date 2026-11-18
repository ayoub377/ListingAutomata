# Meta Catalog bulk formats

Two output files, same data. Field names differ slightly between the Batch API (JSON) and the feed file (CSV) — use exactly the names below.

## Required fields (both formats)

| Concept | JSON (Batch API) | CSV feed | Notes |
|---|---|---|---|
| Unique ID | `retailer_id` | `id` | Stable slug, max 100 chars, no spaces. Reused on updates. |
| Title | `name` | `title` | ≤ 200 chars hard limit; keep ≤ 65 for display. |
| Description | `description` | `description` | ≤ 9999 chars; keep it short. |
| Availability | `availability` | `availability` | `in stock`, `out of stock`, `preorder`, `available for order`, `discontinued`. Default `in stock` unless told otherwise. |
| Condition | `condition` | `condition` | `new`, `refurbished`, `used`. Default `new`. |
| Price | `price` | `price` | JSON: integer in **cents/centimes** + separate `currency` field (e.g. `price: 34900, currency: "MAD"`). CSV: string with currency code, e.g. `349.00 MAD`. |
| Product page | `url` | `link` | Must be a valid URL. Placeholder if user has no site — flag it. |
| Main image | `image_url` | `image_link` | Public direct-image URL only. Default: Google Drive upload → `https://lh3.googleusercontent.com/d/<FILE_ID>` (verified by fetching). Fall back to `UPLOAD://` placeholders if hosting isn't possible. |
| Brand | `brand` | `brand` | From identification; omit if genuinely unbranded (CSV: leave empty). |

## Recommended optional fields

- `additional_image_url` (JSON) / `additional_image_link` (CSV): the two generated angle shots. CSV takes comma-separated URLs in one cell.
- `google_product_category` or `fb_product_category`: improves ad delivery; use the closest standard category string (e.g. `Sporting Goods > Outdoor Recreation > Camping & Hiking > Tents`).
- `sale_price` (same format as price) if the user gives a promo price.
- `inventory`: integer stock count if provided.

## JSON batch payload shape

```json
{
  "allow_upsert": true,
  "requests": [
    {
      "method": "UPDATE",
      "data": {
        "id": "naturehike-cloud-up-2",
        "title": "Naturehike Cloud Up 2 — Ultralight 2-Person Tent",
        "description": "…",
        "availability": "in stock",
        "condition": "new",
        "price": "1499.00 MAD",
        "link": "https://example.com/naturehike-cloud-up-2",
        "image": [
          {"url": "UPLOAD://product-01/01-main.jpg"},
          {"url": "UPLOAD://product-01/02-angle.jpg"},
          {"url": "UPLOAD://product-01/03-context.jpg"}
        ],
        "brand": "Naturehike"
      }
    }
  ]
}
```

This payload is sent to `POST /{catalog_id}/items_batch` (`item_type=PRODUCT_ITEM`) with an access token — the user does that part, or a future automation does. The skill only writes the file.

## CSV feed shape

Header row exactly:

```
id,title,description,availability,condition,price,link,image_link,additional_image_link,brand,google_product_category,item_group_id,color
```

One row per product. Quote any field containing commas. UTF-8, no BOM. This file uploads directly in Commerce Manager → Catalog → Data sources → Data feed → Upload file.

## Colour variants

A product that comes in several colours is one listing per colour, tied together by `item_group_id`
(same field name in the feed and in the Batch API `data` object; `color` likewise).

- `id` stays unique per colour: `chaise-camping-vert`, `chaise-camping-bleu`.
- `item_group_id` is the product's name and is identical on every colour: `chaise-camping`.
- Every listing in a group must fill in `color`. Meta calls a "parent" row with an empty colour the
  wrong way to set variants up, so `MetaListing` refuses a grouped listing without one.
- Title is the same on every colour (Meta's own example does this, so the name does not change when a
  shopper picks a colour); the colour lives in the `color` field and in each variant's photos.
- A single-colour product may send `color` with no `item_group_id`.
- Limits: `item_group_id` 100 chars; `color` 100 (Batch API) / 200 (feed), so 100 is used.

Source: Meta's product-variants guide and catalog reference, checked 2026-11-18. Meta's Batch API page
lists `title` at 100 and `description` at 5000 characters, while the feed page says 200 and 9999; this
project enforces the feed limits.

In `products.csv` this is the `colors` column (`vert; bleu; rouge`), plus optional `photo_color`.

## Validation before writing

- Every `id` unique and slug-safe (`[a-z0-9-]`).
- Every price parses (or is the literal `PRICE_TBD`, which the summary must flag).
- Titles ≤ 200 chars, no leading/trailing whitespace.
- Availability/condition values are from the allowed lists verbatim (lowercase).
- Count of rows in CSV == count of requests in JSON == count of product folders.
