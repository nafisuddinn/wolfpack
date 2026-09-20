import { createClient } from "@supabase/supabase-js";
import type { Database } from "./database.types";

// Read-only, anon/publishable key client. This is the ONLY Supabase client
// the Next.js app should ever construct — it must never see
// SUPABASE_SERVICE_ROLE_KEY, which is reserved for the Python worker's
// server-side writes (see worker/src/wolfpack_worker/db.py).
//
// These are read server-side (server components / route handlers), so they
// intentionally do NOT use the NEXT_PUBLIC_ prefix — matches the naming
// already established in .env.example for SUPABASE_URL / SUPABASE_ANON_KEY.
// If a future task needs this client in a browser/client component, add
// NEXT_PUBLIC_-prefixed equivalents at that point rather than widening the
// exposure of these here.
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error("Missing SUPABASE_URL or SUPABASE_ANON_KEY env vars.");
}

export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: false,
  },
});
