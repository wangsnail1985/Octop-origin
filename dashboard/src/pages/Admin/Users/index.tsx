import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Navigate, useSearchParams } from "react-router-dom";
import { Group, IdCard, Users } from "lucide-react";
import PageShell from "../../../layouts/PageShell";
import TabBar, { type TabBarItem } from "../../../components/TabLabel/TabBar";
import RolesPanel from "./RolesPanel";
import GroupsPanel from "./GroupsPanel";
import UsersListPanel from "./UsersListPanel";
import ForbiddenPage from "../../../components/ForbiddenPage";
import { useGatedSearchTabs } from "../../../hooks/useGatedSearchTabs";
import { USERS_TAB_PERMISSIONS } from "../../../utils/permissions";

type TabKey = "local" | "roles" | "groups";

const TABS: TabBarItem<TabKey>[] = [
  { key: "local", labelKey: "adminUsers.tabLocal", icon: Users },
  { key: "roles", labelKey: "adminUsers.tabRoles", icon: IdCard },
  { key: "groups", labelKey: "adminUsers.tabGroups", icon: Group },
];

function parseTab(raw: string | null): TabKey {
  if (raw === "roles") return "roles";
  if (raw === "groups") return "groups";
  return "local";
}

export default function AdminUsersPage() {
  const [searchParams] = useSearchParams();
  if (searchParams.get("tab") === "config") {
    return <Navigate to="/admin/advanced?tab=config" replace />;
  }
  return <AdminUsersPageBody />;
}

function AdminUsersPageBody() {
  const { t } = useTranslation();
  const { allowedTabs, activeTab, forbidden, selectTab } = useGatedSearchTabs({
    tabs: TABS,
    tabPermissions: USERS_TAB_PERMISSIONS,
    parseTab,
    querylessKey: "local",
  });

  if (forbidden) return <ForbiddenPage />;

  let body: ReactNode = <UsersListPanel />;
  if (activeTab === "roles") {
    body = <RolesPanel />;
  } else if (activeTab === "groups") {
    body = <GroupsPanel />;
  }

  return (
    <PageShell.Tabbed
      title={t("pageShell.adminUsers.title")}
      subtitle={t("pageShell.adminUsers.subtitle")}
      tabBar={
        <TabBar tabs={allowedTabs} activeKey={activeTab} onChange={selectTab} />
      }
    >
      {body}
    </PageShell.Tabbed>
  );
}
