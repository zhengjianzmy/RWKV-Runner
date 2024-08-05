import { FC, lazy, LazyExoticComponent, ReactElement } from 'react';
import {
  ArrowDownload20Regular,
  Chat20Regular,
  ClipboardEdit20Regular,
  DataUsageSettings20Regular,
  DocumentSettings20Regular,
  Home20Regular,
  Info20Regular,
  MusicNote220Regular,
  Settings20Regular,
  Storage20Regular
} from '@fluentui/react-icons';

type NavigationItem = {
  label: string;
  path: string;
  icon: ReactElement;
  element: LazyExoticComponent<FC>;
  top: boolean;
};

export const pages: NavigationItem[] = [
  {
    label: 'Login',
    path: '/login',
    icon: <Home20Regular />,
    element: lazy(() => import('./Login')),
    top: true
  },
  {
    label: 'Chat',
    path: '/chat',
    icon: <Chat20Regular />,
    element: lazy(() => import('./Chat')),
    top: true
  },
  {
    label: 'UploadFile',
    path: '/uploadfile',
    icon: <Settings20Regular />,
    element: lazy(() => import('./UploadFile')),
    top: true
  },
  {
    label: 'FileChat',
    path: '/filechat',
    icon: <Chat20Regular />,
    element: lazy(() => import('./FileChat')),
    top: true
  },
  {
    label: 'Setting',
    path: '/setting',
    icon: <Settings20Regular />,
    element: lazy(() => import('./Setting')),
    top: true
  },
  {
    label: 'Profile',
    path: '/profile',
    icon: <Info20Regular />,
    element: lazy(() => import('./Profile')),
    top: true
  },
  {
    label: 'UserSetting',
    path: '/userSetting',
    icon: <DataUsageSettings20Regular />,
    element: lazy(() => import('./UserSetting')),
    top: true
  },
  {
    label: 'Feedback',
    path: '/feedback',
    icon: <DocumentSettings20Regular />,
    element: lazy(() => import('./Feedback')),
    top: true
  },
  {
    label: 'Logout',
    path: '/logout',
    icon: <Settings20Regular />,
    element: lazy(() => import('./Logout')),
    top: true
  }
];
