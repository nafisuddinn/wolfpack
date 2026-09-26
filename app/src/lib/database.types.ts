export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "14.5";
  };
  public: {
    Tables: {
      personas: {
        Row: {
          avatar_url: string | null;
          created_at: string;
          description: string | null;
          entity_kind: string;
          id: string;
          is_active: boolean;
          name: string;
          slug: string;
          strategy_type: string;
          tagline: string;
        };
        Insert: {
          avatar_url?: string | null;
          created_at?: string;
          description?: string | null;
          entity_kind: string;
          id?: string;
          is_active?: boolean;
          name: string;
          slug: string;
          strategy_type: string;
          tagline: string;
        };
        Update: {
          avatar_url?: string | null;
          created_at?: string;
          description?: string | null;
          entity_kind?: string;
          id?: string;
          is_active?: boolean;
          name?: string;
          slug?: string;
          strategy_type?: string;
          tagline?: string;
        };
        Relationships: [];
      };
      prices: {
        Row: {
          close: number;
          high: number;
          ingested_at: string;
          low: number;
          open: number;
          source: string;
          ticker: string;
          timeframe: string;
          ts: string;
          volume: number | null;
        };
        Insert: {
          close: number;
          high: number;
          ingested_at?: string;
          low: number;
          open: number;
          source?: string;
          ticker: string;
          timeframe?: string;
          ts: string;
          volume?: number | null;
        };
        Update: {
          close?: number;
          high?: number;
          ingested_at?: string;
          low?: number;
          open?: number;
          source?: string;
          ticker?: string;
          timeframe?: string;
          ts?: string;
          volume?: number | null;
        };
        Relationships: [];
      };
      trades: {
        Row: {
          broker: string;
          broker_order_id: string | null;
          client_order_id: string | null;
          fill_price: number | null;
          filled_at: string | null;
          id: string;
          persona_id: string;
          qty: number;
          rationale: string | null;
          rationale_author: string | null;
          rationale_written_at: string | null;
          run_id: string;
          side: string;
          signal_payload: Json;
          signal_ts: string;
          status: string;
          submitted_at: string;
          ticker: string;
        };
        Insert: {
          broker?: string;
          broker_order_id?: string | null;
          client_order_id?: string | null;
          fill_price?: number | null;
          filled_at?: string | null;
          id?: string;
          persona_id: string;
          qty: number;
          rationale?: string | null;
          rationale_author?: string | null;
          rationale_written_at?: string | null;
          run_id: string;
          side: string;
          signal_payload?: Json;
          signal_ts: string;
          status?: string;
          submitted_at?: string;
          ticker: string;
        };
        Update: {
          broker?: string;
          broker_order_id?: string | null;
          client_order_id?: string | null;
          fill_price?: number | null;
          filled_at?: string | null;
          id?: string;
          persona_id?: string;
          qty?: number;
          rationale?: string | null;
          rationale_author?: string | null;
          rationale_written_at?: string | null;
          run_id?: string;
          side?: string;
          signal_payload?: Json;
          signal_ts?: string;
          status?: string;
          submitted_at?: string;
          ticker?: string;
        };
        Relationships: [
          {
            foreignKeyName: "trades_persona_id_fkey";
            columns: ["persona_id"];
            isOneToOne: false;
            referencedRelation: "personas";
            referencedColumns: ["id"];
          },
        ];
      };
      trust_weights: {
        Row: {
          computed_at: string;
          id: string;
          inputs: Json;
          method_version: string;
          persona_id: string;
          social_score: number | null;
          statistical_score: number | null;
          week_start: string;
          weight: number;
        };
        Insert: {
          computed_at?: string;
          id?: string;
          inputs?: Json;
          method_version?: string;
          persona_id: string;
          social_score?: number | null;
          statistical_score?: number | null;
          week_start: string;
          weight: number;
        };
        Update: {
          computed_at?: string;
          id?: string;
          inputs?: Json;
          method_version?: string;
          persona_id?: string;
          social_score?: number | null;
          statistical_score?: number | null;
          week_start?: string;
          weight?: number;
        };
        Relationships: [
          {
            foreignKeyName: "trust_weights_persona_id_fkey";
            columns: ["persona_id"];
            isOneToOne: false;
            referencedRelation: "personas";
            referencedColumns: ["id"];
          },
        ];
      };
    };
    Views: {
      feed_trades: {
        Row: {
          broker: string | null;
          broker_order_id: string | null;
          fill_price: number | null;
          filled_at: string | null;
          id: string | null;
          persona_entity_kind: string | null;
          persona_id: string | null;
          persona_name: string | null;
          persona_slug: string | null;
          persona_strategy_type: string | null;
          qty: number | null;
          rationale: string | null;
          rationale_author: string | null;
          rationale_written_at: string | null;
          run_id: string | null;
          side: string | null;
          signal_payload: Json | null;
          signal_ts: string | null;
          status: string | null;
          submitted_at: string | null;
          ticker: string | null;
        };
        Relationships: [
          {
            foreignKeyName: "trades_persona_id_fkey";
            columns: ["persona_id"];
            isOneToOne: false;
            referencedRelation: "personas";
            referencedColumns: ["id"];
          },
        ];
      };
      price_returns: {
        Row: {
          close: number | null;
          log_return: number | null;
          ticker: string | null;
          timeframe: string | null;
          ts: string | null;
        };
        Relationships: [];
      };
    };
    Functions: {
      set_trade_rationale: {
        Args: { p_rationale: string; p_trade_id: string };
        Returns: undefined;
      };
    };
    Enums: {
      [_ in never]: never;
    };
    CompositeTypes: {
      [_ in never]: never;
    };
  };
};

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">;

type DefaultSchema = DatabaseWithoutInternals[Extract<
  keyof Database,
  "public"
>];

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends (DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals;
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R;
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R;
      }
      ? R
      : never
    : never;

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    keyof DefaultSchema["Tables"] | { schema: keyof DatabaseWithoutInternals },
  TableName extends (DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals;
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I;
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I;
      }
      ? I
      : never
    : never;

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    keyof DefaultSchema["Tables"] | { schema: keyof DatabaseWithoutInternals },
  TableName extends (DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals;
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never) = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U;
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U;
      }
      ? U
      : never
    : never;

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    keyof DefaultSchema["Enums"] | { schema: keyof DatabaseWithoutInternals },
  EnumName extends (DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals;
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never) = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never;

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends (PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals;
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never) = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals;
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never;

export const Constants = {
  public: {
    Enums: {},
  },
} as const;
