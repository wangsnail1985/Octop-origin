import { request, requestUpload } from "../request";

export interface GroupAccessItem {
  key: string;
  label: string;
  missing: boolean;
  audience: "all" | "groups" | "creator" | "owner";
}

export interface GroupAccessSummary {
  /** Per-instance grants (legacy feature-wide enum is no longer returned). */
  connectors: GroupAccessItem[];
  storage: GroupAccessItem[];
  models: GroupAccessItem[];
  plugins: GroupAccessItem[];
  skill_packages: GroupAccessItem[];
  knowledge_bases: GroupAccessItem[];
  bridges: GroupAccessItem[];
  experts: GroupAccessItem[];
}

export interface UserGroup {
  user_group_id: string;
  name: string;
  description: string | null;
  enabled: boolean;
  avatar_icon?: string | null;
  avatar_url?: string | null;
  member_count: number;
  member_ids?: number[];
  access?: GroupAccessSummary;
  created_at: number;
  updated_at: number;
}

export interface UserGroupWrite {
  name?: string;
  user_group_id?: string | null;
  description?: string | null;
  enabled?: boolean;
  avatar_icon?: string | null;
}

export type ResourceVisibilityMode = "all" | "groups" | "owner" | "department";

/** Resources whose API accepts the picker types except the KB-only department mode. */
export type SharedVisibilityMode = Exclude<ResourceVisibilityMode, "department">;

export function isSharedVisibility(
  value: ResourceVisibilityMode,
): value is SharedVisibilityMode {
  return value !== "department";
}

export interface SharingAudience {
  user_count: number;
  group_count: number;
  /** Group projected from the caller's role, when it exists (department share). */
  my_sync_group_id?: string | null;
}

export const userGroupsApi = {
  list: () => request<UserGroup[]>("/user-groups"),

  /** Enabled accounts and user groups. Any signed-in user may call this. */
  audience: () => request<SharingAudience>("/user-groups/audience"),

  create: (body: UserGroupWrite & { name: string }) =>
    request<UserGroup>("/user-groups", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  update: (userGroupId: string, body: UserGroupWrite) =>
    request<UserGroup>(`/user-groups/${encodeURIComponent(userGroupId)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  uploadAvatar: (userGroupId: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return requestUpload<{ avatar_url: string | null }>(
      `/user-groups/${encodeURIComponent(userGroupId)}/avatar`,
      body,
    );
  },

  deleteAvatar: (userGroupId: string) =>
    request<void>(`/user-groups/${encodeURIComponent(userGroupId)}/avatar`, {
      method: "DELETE",
    }),

  remove: (userGroupId: string) =>
    request<void>(`/user-groups/${encodeURIComponent(userGroupId)}`, {
      method: "DELETE",
    }),

  members: (userGroupId: string) =>
    request<{ user_ids: number[] }>(
      `/user-groups/${encodeURIComponent(userGroupId)}/members`,
    ),

  replaceMembers: (userGroupId: string, userIds: number[]) =>
    request<{ user_ids: number[] }>(
      `/user-groups/${encodeURIComponent(userGroupId)}/members`,
      {
        method: "PUT",
        body: JSON.stringify({ user_ids: userIds }),
      },
    ),
};
